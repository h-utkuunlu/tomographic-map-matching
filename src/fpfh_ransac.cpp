#include <pcl/common/transforms.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <spdlog/spdlog.h>
#include <tomographic_map_matching/fpfh_ransac.hpp>

namespace map_matcher {

void to_json(json &j, const FPFHRANSACParameters &p) {
  to_json(j, static_cast<Parameters>(p));
  j["normal_radius"] = p.normal_radius;
  j["descriptor_radius"] = p.descriptor_radius;
  j["response_method"] = p.response_method;
  j["keypoint_radius"] = p.keypoint_radius;
  j["corner_threshold"] = p.corner_threshold;
  j["ransac_inlier_threshold"] = p.ransac_inlier_threshold;
  j["ransac_refine_model"] = p.ransac_refine_model;
}

void from_json(const json &j, FPFHRANSACParameters &p) {
  Parameters p_base;
  from_json(j, p_base);
  p = FPFHRANSACParameters(p_base);
  if (j.contains("normal_radius"))
    j.at("normal_radius").get_to(p.normal_radius);

  if (j.contains("descriptor_radius"))
    j.at("descriptor_radius").get_to(p.descriptor_radius);

  if (j.contains("keypoint_radius"))
    j.at("keypoint_radius").get_to(p.keypoint_radius);

  if (j.contains("response_method"))
    j.at("response_method").get_to(p.response_method);

  if (j.contains("corner_threshold"))
    j.at("corner_threshold").get_to(p.corner_threshold);

  if (j.contains("ransac_inlier_threshold"))
    j.at("ransac_inlier_threshold").get_to(p.ransac_inlier_threshold);

  if (j.contains("ransac_refine_model"))
    j.at("ransac_refine_model").get_to(p.ransac_refine_model);
}

FPFHRANSAC::FPFHRANSAC() : MapMatcherBase() {}

FPFHRANSAC::FPFHRANSAC(FPFHRANSACParameters parameters)
    : MapMatcherBase(static_cast<Parameters>(parameters)),
      parameters_(parameters) {
  if (parameters_.response_method < 1 or parameters_.response_method > 5) {
    spdlog::warn("Corner response method must be in the range 1-5 (Harris, "
                 "Noble, Lowe, Tomasi, Curvature). Defaulting to Harris");
    parameters_.response_method = 1;
  }
}

json FPFHRANSAC::GetParameters() const {
  json retval = parameters_;
  return retval;
}

void FPFHRANSAC::SetParameters(const json &parameters) {
  parameters_ = parameters.template get<FPFHRANSACParameters>();

  if (parameters_.response_method < 1 or parameters_.response_method > 5) {
    spdlog::warn("Corner response method must be in the range 1-5 (Harris, "
                 "Noble, Lowe, Tomasi, Curvature). Defaulting to Harris");
    parameters_.response_method = 1;
  }
}

void FPFHRANSAC::DetectAndDescribeKeypoints(const PointCloud::Ptr input,
                                            PointCloud::Ptr keypoints,
                                            FeatureCloud::Ptr features) const {

  spdlog::debug("PCD size: {}", input->size());

  std::chrono::steady_clock::time_point timer;
  timer = std::chrono::steady_clock::now();

  // Normals are needed for both keypoints and the FPFH
  NormalCloud::Ptr normals(new NormalCloud);
  pcl::NormalEstimationOMP<PointT, NormalT> normal_estimator;
  normal_estimator.setRadiusSearch(parameters_.normal_radius);
  normal_estimator.setInputCloud(input);
  normal_estimator.compute(*normals);

  spdlog::debug("Normal estimation took {} s", CalculateTimeSince(timer));
  timer = std::chrono::steady_clock::now();

  // Keypoints
  KeypointCloud::Ptr keypoints_with_response(new KeypointCloud);
  KeypointDetector detector;
  auto response_method = static_cast<KeypointDetector::ResponseMethod>(
      parameters_.response_method);
  detector.setMethod(response_method);
  detector.setRadius(parameters_.keypoint_radius);
  detector.setThreshold(parameters_.corner_threshold);
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
  descriptor.setRadiusSearch(parameters_.descriptor_radius);
  descriptor.setInputCloud(keypoints);
  descriptor.setSearchSurface(input);
  descriptor.setInputNormals(normals);
  descriptor.setSearchMethod(normal_estimator.getSearchMethod());
  descriptor.compute(*features);

  spdlog::debug("Feature computation took {} s", CalculateTimeSince(timer));
}

void FPFHRANSAC::ExtractInlierKeypoints(
    const PointCloud::Ptr map1_pcd, const PointCloud::Ptr map2_pcd,
    const pcl::CorrespondencesPtr correspondences, PointCloud::Ptr map1_inliers,
    PointCloud::Ptr map2_inliers) const {

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

HypothesisPtr FPFHRANSAC::RegisterPointCloudMaps(const PointCloud::Ptr map1_pcd,
                                                 const PointCloud::Ptr map2_pcd,
                                                 json &stats) const {

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
  rejector_ransac.setInlierThreshold(parameters_.ransac_inlier_threshold);
  rejector_ransac.setRefineModel(parameters_.ransac_refine_model);

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
  ExtractInlierKeypoints(map1_keypoints, map2_keypoints, correspondences_inlier,
                         map1_inliers, map2_inliers);

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
  result->pose = ConstructTransformFromParameters(result->x, result->y,
                                                  result->z, result->theta);

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

void FPFHRANSAC::VisualizeKeypoints(const PointCloud::Ptr points,
                                    const PointCloud::Ptr keypoints) const {

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
