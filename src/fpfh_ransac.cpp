#include <pcl/common/transforms.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <tomographic_map_matching/fpfh_ransac.hpp>

namespace map_matcher {

FPFHRANSAC::FPFHRANSAC()
  : MapMatcherBase()
{
}

FPFHRANSAC::FPFHRANSAC(const json& parameters)
  : MapMatcherBase(parameters)
{
  UpdateParameters(parameters);
}

void
FPFHRANSAC::GetParameters(json& output) const
{
  MapMatcherBase::GetParameters(output);
  output["normal_radius"] = normal_radius_;
  output["descriptor_radius"] = descriptor_radius_;
  output["keypoint_radius"] = keypoint_radius_;
  output["response_method"] = response_method_;
  output["corner_threshold"] = corner_threshold_;
  output["ransac_inlier_threshold"] = ransac_inlier_threshold_;
  output["ransac_refine_model"] = ransac_refine_model_;
}

void
FPFHRANSAC::UpdateParameters(const json& input)
{

  UpdateSingleParameter(input, "normal_radius", normal_radius_);
  UpdateSingleParameter(input, "descriptor_radius", descriptor_radius_);
  UpdateSingleParameter(input, "keypoint_radius", keypoint_radius_);

  UpdateSingleParameter(input, "response_method", response_method_);

  if (response_method_ < 1 or response_method_ > 5) {
    spdlog::warn("Corner response method must be in the range 1-5 (Harris, "
                 "Noble, Lowe, Tomasi, Curvature). Defaulting to Harris");
    response_method_ = 1;
  }

  UpdateSingleParameter(input, "corner_threshold", corner_threshold_);
  UpdateSingleParameter(
    input, "ransac_inlier_threshold", ransac_inlier_threshold_);
  UpdateSingleParameter(input, "ransac_refine_model", ransac_refine_model_);
}

void
FPFHRANSAC::DetectAndDescribeKeypoints(const PointCloud::Ptr input,
                                       PointCloud::Ptr keypoints,
                                       FeatureCloud::Ptr features) const
{

  spdlog::debug("PCD size: {}", input->size());

  std::chrono::steady_clock::time_point timer;
  timer = std::chrono::steady_clock::now();

  // Normals are needed for both keypoints and the FPFH
  NormalCloud::Ptr normals(new NormalCloud);
  pcl::NormalEstimationOMP<PointT, NormalT> normal_estimator;
  normal_estimator.setRadiusSearch(normal_radius_);
  normal_estimator.setInputCloud(input);
  normal_estimator.compute(*normals);

  spdlog::debug("Normal estimation took {} s", CalculateTimeSince(timer));
  timer = std::chrono::steady_clock::now();

  // Keypoints
  KeypointCloud::Ptr keypoints_with_response(new KeypointCloud);
  KeypointDetector detector;
  auto response_method =
    static_cast<KeypointDetector::ResponseMethod>(response_method_);
  detector.setMethod(response_method);
  detector.setRadius(keypoint_radius_);
  detector.setThreshold(corner_threshold_);
  detector.setInputCloud(input);
  detector.setNormals(normals);
  detector.setSearchMethod(normal_estimator.getSearchMethod());
  detector.compute(*keypoints_with_response);

  // Extract XYZ only. Output to "keypoints"
  pcl::ExtractIndices<PointT> selector;
  selector.setInputCloud(input);
  selector.setIndices(detector.getKeypointsIndices());
  selector.filter(*keypoints);

  spdlog::debug("Keypoint detection took {} s", CalculateTimeSince(timer));
  spdlog::debug("Num. keypoints: {}", keypoints->size());
  timer = std::chrono::steady_clock::now();

  // Calculate FPFH for the keypoints. Output to "features"
  pcl::FPFHEstimationOMP<PointT, NormalT, FeatureT> descriptor;
  descriptor.setRadiusSearch(descriptor_radius_);
  descriptor.setInputCloud(keypoints);
  descriptor.setSearchSurface(input);
  descriptor.setInputNormals(normals);
  descriptor.setSearchMethod(normal_estimator.getSearchMethod());
  descriptor.compute(*features);

  spdlog::debug("Feature computation took {} s", CalculateTimeSince(timer));
}

void
FPFHRANSAC::ExtractInlierKeypoints(
  const PointCloud::Ptr map1_pcd,
  const PointCloud::Ptr map2_pcd,
  const pcl::CorrespondencesPtr correspondences,
  PointCloud::Ptr map1_inliers,
  PointCloud::Ptr map2_inliers) const
{

  // The assumption here is that the map1_pcd is the target (match), map2_pcd is
  // the source (query)
  size_t N = correspondences->size();
  map1_inliers->resize(N);
  map2_inliers->resize(N);

  for (size_t i = 0; i < N; ++i) {
    map2_inliers->at(i) = map2_pcd->points[correspondences->at(i).index_query];
    map1_inliers->at(i) = map1_pcd->points[correspondences->at(i).index_match];
  }
}

HypothesisPtr
FPFHRANSAC::RegisterPointCloudMaps(const PointCloud::Ptr map1_pcd,
                                   const PointCloud::Ptr map2_pcd,
                                   json& stats) const
{

  if (map1_pcd->size() == 0 or map2_pcd->size() == 0) {
    spdlog::critical("Pointcloud(s) are empty. Aborting");
    return HypothesisPtr(new Hypothesis());
  }

  // Timing
  std::chrono::steady_clock::time_point total, indiv;
  total = std::chrono::steady_clock::now();
  indiv = std::chrono::steady_clock::now();

  // Compute keypoints and features
  PointCloud::Ptr map1_keypoints(new PointCloud),
    map2_keypoints(new PointCloud);
  FeatureCloud::Ptr map1_features(new FeatureCloud),
    map2_features(new FeatureCloud);

  DetectAndDescribeKeypoints(map1_pcd, map1_keypoints, map1_features);
  DetectAndDescribeKeypoints(map2_pcd, map2_keypoints, map2_features);

  stats["t_feature_extraction"] = CalculateTimeSince(indiv);
  stats["map1_num_features"] = map1_features->size();
  stats["map2_num_features"] = map2_features->size();

  spdlog::debug("Feature extraction took {} s",
                stats["t_feature_extraction"].template get<double>());
  indiv = std::chrono::steady_clock::now();

  // Matching & registration
  pcl::registration::CorrespondenceEstimation<FeatureT, FeatureT>
    correspondence_estimator;
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
  correspondence_estimator.setInputSource(map2_features);
  correspondence_estimator.setInputTarget(map1_features);
  correspondence_estimator.determineCorrespondences(*correspondences);
  spdlog::debug("Matching complete");

  // Limit to one-to-one matches
  pcl::CorrespondencesPtr correspondences_one_to_one(new pcl::Correspondences);
  pcl::registration::CorrespondenceRejectorOneToOne rejector_one_to_one;
  rejector_one_to_one.setInputCorrespondences(correspondences);
  rejector_one_to_one.getCorrespondences(*correspondences_one_to_one);
  spdlog::debug("One-to-one rejection complete");

  // Correspondance rejection with RANSAC
  pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>
    rejector_ransac;
  pcl::CorrespondencesPtr correspondences_inlier(new pcl::Correspondences);
  rejector_ransac.setInlierThreshold(ransac_inlier_threshold_);
  rejector_ransac.setRefineModel(ransac_refine_model_);

  rejector_ransac.setInputSource(map2_keypoints);
  rejector_ransac.setInputTarget(map1_keypoints);
  rejector_ransac.setInputCorrespondences(correspondences_one_to_one);
  rejector_ransac.getCorrespondences(*correspondences_inlier);
  Eigen::Matrix4f transform = rejector_ransac.getBestTransformation();

  stats["t_pose_estimation"] = CalculateTimeSince(indiv);
  spdlog::debug("Pose estimation completed in {}. Num. inliers: {}",
                stats["t_pose_estimation"].template get<double>(),
                correspondences_inlier->size());

  // Extract inliers
  PointCloud::Ptr map1_inliers(new PointCloud), map2_inliers(new PointCloud);
  ExtractInlierKeypoints(map1_keypoints,
                         map2_keypoints,
                         correspondences_inlier,
                         map1_inliers,
                         map2_inliers);

  // Construct result
  HypothesisPtr result(new Hypothesis);
  result->n_inliers = correspondences_inlier->size();
  result->x = transform(0, 3);
  result->y = transform(1, 3);
  result->z = transform(2, 3);

  Eigen::Matrix3f rotm = transform.block<3, 3>(0, 0);
  Eigen::AngleAxisf axang(rotm);
  float angle = axang.angle() * axang.axis()(2);
  result->theta = angle;
  result->pose = ConstructTransformFromParameters(
    result->x, result->y, result->z, result->theta);

  result->inlier_points_1 = map1_inliers;
  result->inlier_points_2 = map2_inliers;

  stats["t_total"] = CalculateTimeSince(total);

  // Measure memory use
  stats["mem_cpu"] = GetPeakRSS();

  // // Visualize features
  // VisualizeKeypoints(map1_pcd, map1_kp_coords);
  // VisualizeKeypoints(map2_pcd, map2_kp_coords);

  return result;
}

void
FPFHRANSAC::VisualizeKeypoints(const PointCloud::Ptr points,
                               const PointCloud::Ptr keypoints) const
{

  pcl::visualization::PCLVisualizer viewer("Keypoints");
  PointCloudColor points_color(points, 155, 0, 0),
    keypoints_color(keypoints, 0, 155, 0);

  viewer.addPointCloud(points, points_color, "points");
  viewer.addPointCloud(keypoints, keypoints_color, "keypoints");

  viewer.setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");

  viewer.spin();
}

} // namespace map_matcher
