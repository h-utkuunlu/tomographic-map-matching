#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <sys/resource.h>
#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

MapMatcherBase::MapMatcherBase() {}

MapMatcherBase::MapMatcherBase(const json& parameters)
{
  UpdateParameters(parameters);
}

void
MapMatcherBase::UpdateParameters(const json& input)
{
  UpdateSingleParameter(input, "algorithm", algorithm_);
  UpdateSingleParameter(input, "grid_size", grid_size_);
  UpdateSingleParameter(input, "icp_refinement", icp_refinement_);
};

void
MapMatcherBase::GetParameters(json& output) const
{
  output["algorithm"] = algorithm_;
  output["grid_size"] = grid_size_;
  output["icp_refinement"] = icp_refinement_;
};

size_t
MapMatcherBase::GetPeakRSS() const
{
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return (size_t)(rusage.ru_maxrss * 1024L);
}

double
MapMatcherBase::CalculateTimeSince(
  const std::chrono::steady_clock::time_point& start) const
{
  return std::chrono::duration_cast<std::chrono::microseconds>(
           std::chrono::steady_clock::now() - start)
           .count() /
         1000000.0;
}

CartesianBounds
MapMatcherBase::CalculateBounds(const PointCloud::Ptr& pcd) const
{
  auto low = std::numeric_limits<float>::lowest(),
       high = std::numeric_limits<float>::max();
  PointT upper(low, low, low), lower(high, high, high);

  for (const auto& pt : *pcd) {
    if (pt.x > upper.x) {
      upper.x = pt.x;
    }
    if (pt.y > upper.y) {
      upper.y = pt.y;
    }
    if (pt.z > upper.z) {
      upper.z = pt.z;
    }

    if (pt.x < lower.x) {
      lower.x = pt.x;
    }
    if (pt.y < lower.y) {
      lower.y = pt.y;
    }
    if (pt.z < lower.z) {
      lower.z = pt.z;
    }
  }
  return CartesianBounds(upper, lower);
}

PointT
MapMatcherBase::CalculateXYZSpread(const PointCloud::Ptr& pcd) const
{
  double N = static_cast<double>(pcd->size());

  // Since we know the maps to be gravity aligned, we are only interested in the
  // spread along the xy plane. Using PCA to extract the major axes. For Z axis,
  // we only need the simple std. deviation estimate
  PointT mean(0.0, 0.0, 0.0);
  double z_sq = 0.0;
  Eigen::Matrix<double, 2, Eigen::Dynamic> X(2, pcd->size());

  // Accumulate matrix terms, means and z squared values
  size_t idx = 0;
  for (const PointT& pt : pcd->points) {
    mean.x += pt.x;
    mean.y += pt.y;
    mean.z += pt.z;
    X.col(idx++) << pt.x, pt.y;
    z_sq += pt.z * pt.z;
  }

  mean.x /= N;
  mean.y /= N;
  mean.z /= N;

  // For Z spread, using sample stdev formula with finite population
  double spread_z = std::sqrt(z_sq / N - (mean.z * mean.z));

  // For XY spread, perform SVD with zero mean
  X = X.colwise() - Eigen::Vector2d(mean.x, mean.y);
  Eigen::JacobiSVD<Eigen::Matrix<double, 2, Eigen::Dynamic>> svd(
    X, Eigen::ComputeThinU | Eigen::ComputeThinV);

  auto sv = svd.singularValues();

  double sqrt_n = std::sqrt(N);
  double spread_1 = sv[0] / sqrt_n, spread_2 = sv[1] / sqrt_n;

  return PointT(spread_1, spread_2, spread_z);
}

Eigen::Matrix4d
MapMatcherBase::ConstructTransformFromParameters(double x,
                                                 double y,
                                                 double z,
                                                 double t) const
{
  Eigen::Affine3d result_pose = Eigen::Affine3d::Identity();
  result_pose.translation() << x, y, z;
  result_pose.rotate(Eigen::AngleAxisd(t, Eigen::Vector3d::UnitZ()));
  Eigen::Matrix4d result_pose_mat = result_pose.matrix();
  return result_pose_mat;
}

void
MapMatcherBase::VisualizeHypothesis(const PointCloud::Ptr& map1_pcd,
                                    const PointCloud::Ptr& map2_pcd,
                                    const HypothesisPtr& result) const
{
  // 3 windows: Map 1, Map2, and merged maps
  pcl::visualization::PCLVisualizer viewer("Results");
  int vp0 = 0, vp1 = 1, vp2 = 2;
  viewer.createViewPort(0.0, 0.5, 0.5, 1.0, vp0);
  viewer.createViewPort(0.5, 0.5, 1.0, 1.0, vp1);
  viewer.createViewPort(0.0, 0.0, 1.0, 0.5, vp2);

  // Before merging
  PointCloudColorGF map1_color(map1_pcd, "z"), map2_color(map2_pcd, "z");
  viewer.addPointCloud(map1_pcd, map1_color, "map1", vp0);
  viewer.addPointCloud(map2_pcd, map2_color, "map2", vp1);

  // After merging
  PointCloud::Ptr map2_transformed(new PointCloud);
  pcl::transformPointCloud(*map2_pcd, *map2_transformed, result->pose);

  PointCloudColor map1_merged_color(map1_pcd, 155, 0, 0),
    map2_merged_color(map2_transformed, 0, 155, 0);
  viewer.addPointCloud(map1_pcd, map1_merged_color, "map1_merged", vp2);
  viewer.addPointCloud(map2_transformed, map2_merged_color, "map2_merged", vp2);

  // Draw inliers if they are availabile
  if (result->inlier_points_1 != nullptr) {
    PointCloud::Ptr map2_inliers_transformed(new PointCloud);
    pcl::transformPointCloud(
      *(result->inlier_points_2), *map2_inliers_transformed, result->pose);

    PointCloudColor map1_inliers_color(result->inlier_points_1, 255, 0, 255),
      map2_inliers_color(map2_inliers_transformed, 0, 255, 255);

    viewer.addPointCloud(
      result->inlier_points_1, map1_inliers_color, "map1_inliers", vp2);
    viewer.addPointCloud(
      map2_inliers_transformed, map2_inliers_color, "map2_inliers", vp2);
    viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "map1_inliers");
    viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "map2_inliers");

    // Add lines between all correspondences
    for (size_t i = 0; i < result->inlier_points_1->size(); ++i) {
      const PointT &pt1 = result->inlier_points_1->points.at(i),
                   &pt2 = map2_inliers_transformed->points.at(i);
      std::string line_name = "inlier_line_" + std::to_string(i + 1);
      viewer.addLine(pt1, pt2, 255, 255, 255, line_name, vp2);
    }
  }

  viewer.spin();
}

PointT
MapMatcherBase::ComputeResultSpread(const HypothesisPtr& result) const
{
  // Construct inliers pcds if they do not exist
  if (result->inlier_points_1 == nullptr ||
      result->inlier_points_2 == nullptr) {
    result->inlier_points_1 = PointCloud::Ptr(new PointCloud());
    result->inlier_points_2 = PointCloud::Ptr(new PointCloud());

    for (const SliceTransformPtr& tf_result : result->inlier_slices) {
      // Retrieve respective slice heights
      double height_map1 = tf_result->slice1->height,
             height_map2 = tf_result->slice2->height;

      for (const std::pair<cv::Point2f, cv::Point2f>& inlier_pair :
           tf_result->inliers) {
        const auto &pt1 = inlier_pair.first, &pt2 = inlier_pair.second;

        result->inlier_points_1->push_back(PointT(pt1.x, pt1.y, height_map1));
        result->inlier_points_2->push_back(PointT(pt2.x, pt2.y, height_map2));
      }
    }
  }

  // Calculate spreads for each map
  PointT spread_map1 = CalculateXYZSpread(result->inlier_points_1),
         spread_map2 = CalculateXYZSpread(result->inlier_points_2);

  // Return minimum instead of average. Should be a more conservative estimate
  // The spreads should actually be the same in z, and pretty close in x and y
  return PointT(std::min(spread_map1.x, spread_map2.x),
                std::min(spread_map1.y, spread_map2.y),
                std::min(spread_map1.z, spread_map2.z));
}

HypothesisPtr
MapMatcherBase::RefineResult(const HypothesisPtr& result) const
{
  HypothesisPtr result_refined(new Hypothesis(*result));
  Eigen::Matrix4d result_pose = result->pose;

  // ICP to be performed over feature points rather than all points
  PointCloud::Ptr inliers2_refinement(new PointCloud());
  pcl::transformPointCloud(
    *(result->inlier_points_2), *inliers2_refinement, result_pose);
  Eigen::Matrix4f icp_result_f = Eigen::Matrix4f::Identity();

  if (icp_refinement_ == 1) { // ICP
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputTarget(result->inlier_points_1);
    icp.setInputSource(inliers2_refinement);
    icp.setMaxCorrespondenceDistance(grid_size_ * 2.0);
    icp.setUseReciprocalCorrespondences(true);
    PointCloud::Ptr resIcp(new PointCloud());
    icp.align(*resIcp);
    icp_result_f = icp.getFinalTransformation();

  } else if (icp_refinement_ == 2) { // GICP
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> icp;
    icp.setInputTarget(result->inlier_points_1);
    icp.setInputSource(inliers2_refinement);
    icp.setMaxCorrespondenceDistance(grid_size_ * 2.0);
    icp.setUseReciprocalCorrespondences(true);
    PointCloud::Ptr resIcp(new PointCloud());
    icp.align(*resIcp);
    icp_result_f = icp.getFinalTransformation();
  }

  Eigen::Matrix4d icp_result_d = icp_result_f.cast<double>();
  Eigen::Matrix4d solution = icp_result_d * result_pose.matrix();
  Eigen::Matrix3d solution_rotm = solution.topLeftCorner(3, 3);
  Eigen::Vector3d euler_solution = solution_rotm.eulerAngles(2, 1, 0);

  result_refined->x = solution(0, 3);
  result_refined->y = solution(1, 3);
  result_refined->z = solution(2, 3);
  result_refined->theta = euler_solution(0);
  result_refined->pose = solution;

  return result_refined;
}

} // namespace map_matcher
