#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/harris_3d.h>
#include <tomographic_map_matching/fpfh_base.hpp>

namespace map_matcher {

FPFHBase::FPFHBase()
  : MapMatcherBase()
{
}

FPFHBase::FPFHBase(const json& parameters)
  : MapMatcherBase(parameters)
{
  UpdateParameters(parameters);
}

void
FPFHBase::GetParameters(json& output) const
{
  MapMatcherBase::GetParameters(output);
  output["normal_radius"] = normal_radius_;
  output["descriptor_radius"] = descriptor_radius_;
  output["keypoint_radius"] = keypoint_radius_;
  output["response_method"] = response_method_;
  output["corner_threshold"] = corner_threshold_;
}

void
FPFHBase::UpdateParameters(const json& input)
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
}

void
FPFHBase::VisualizeKeypoints(const PointCloud::Ptr points,
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

void
FPFHBase::DetectAndDescribeKeypoints(const PointCloud::Ptr input,
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
FPFHBase::ExtractInlierKeypoints(const PointCloud::Ptr map1_pcd,
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

} // namespace map_matcher
