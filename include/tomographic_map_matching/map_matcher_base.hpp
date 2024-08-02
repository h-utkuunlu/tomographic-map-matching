#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <spdlog/spdlog.h>

namespace map_matcher {

using json = nlohmann::json;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> PointCloudColor;
typedef pcl::visualization::PointCloudColorHandlerGenericField<PointT>
  PointCloudColorGF;

struct CartesianBounds
{
  CartesianBounds()
  {
    const double lowest = std::numeric_limits<double>::lowest(),
                 largest = std::numeric_limits<double>::max();
    upper = PointT(lowest, lowest, lowest);
    lower = PointT(largest, largest, largest);
  }
  CartesianBounds(PointT upper_input, PointT lower_input)
  {
    upper = upper_input;
    lower = lower_input;
  }
  PointT upper, lower;
};

struct Slice
{
  Slice()
  {
    height = 0.0;
    binary_image = cv::Mat();
    slice_bounds = CartesianBounds();
    is_descriptive = false;
    kp = std::vector<cv::KeyPoint>();
    desc = cv::Mat();
    matcher = nullptr;
  }

  double height;
  cv::Mat binary_image;
  CartesianBounds slice_bounds;
  bool is_descriptive;
  std::vector<cv::KeyPoint> kp;
  cv::Mat desc;
  cv::Ptr<cv::DescriptorMatcher> matcher;
};
typedef std::shared_ptr<Slice> SlicePtr;

struct SliceTransform
{
  SliceTransform()
  {
    inliers = std::vector<std::pair<cv::Point2f, cv::Point2f>>();
    x = 0.0;
    y = 0.0;
    z = 0.0;
    theta = 0.0;
    pose = Eigen::Matrix4d::Identity();
  }

  std::shared_ptr<Slice> target_slice, source_slice;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> inliers;
  double x, y, z, theta;
  Eigen::Matrix4d pose;
};
typedef std::shared_ptr<SliceTransform> SliceTransformPtr;

struct HeightIndices
{
  size_t m1_min, m1_max, m2_min, m2_max;
};

struct Hypothesis
{
  Hypothesis()
  {
    n_inliers = 0;
    x = 0.0;
    y = 0.0;
    z = 0.0;
    theta = 0.0;
    inlier_slices = std::vector<SliceTransformPtr>();
    inlier_points_1 = nullptr;
    inlier_points_2 = nullptr;
    pose = Eigen::Matrix4d::Identity();
  }

  Hypothesis(const Hypothesis& other)
  {
    n_inliers = other.n_inliers;
    x = other.x;
    y = other.y;
    z = other.z;
    theta = other.theta;
    inlier_slices = other.inlier_slices;
    inlier_points_1 = other.inlier_points_1;
    inlier_points_2 = other.inlier_points_2;
    pose = other.pose;
  }

  size_t n_inliers;
  double x, y, z, theta;
  std::vector<SliceTransformPtr> inlier_slices;
  PointCloud::Ptr inlier_points_1, inlier_points_2;
  Eigen::Matrix4d pose;

  bool operator<(const Hypothesis& rhs) { return n_inliers < rhs.n_inliers; }
};
typedef std::shared_ptr<Hypothesis> HypothesisPtr;

class MapMatcherBase
{
public:
  virtual HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr source,
                                               const PointCloud::Ptr target,
                                               json& stats) const = 0;
  virtual std::string GetName() const = 0;
  virtual void UpdateParameters(const json& input);
  virtual void GetParameters(json& output) const;

  void VisualizeHypothesis(const PointCloud::Ptr& source,
                           const PointCloud::Ptr& target,
                           const HypothesisPtr& result) const;

protected:
  MapMatcherBase();
  MapMatcherBase(const json& parameters);

  template<typename T>
  void UpdateSingleParameter(const json& input,
                             const std::string parameter_name,
                             T& parameter)
  {
    if (input.contains(parameter_name)) {
      input.at(parameter_name).get_to(parameter);
      spdlog::debug("Parameter {} set to {}", parameter_name, parameter);
    }
  }

  size_t GetPeakRSS() const;
  double CalculateTimeSince(const std::chrono::steady_clock::time_point& start) const;
  CartesianBounds CalculateBounds(const PointCloud::Ptr& pcd) const;
  PointT CalculateXYZSpread(const PointCloud::Ptr& pcd) const;
  double EstimateStdDev(const std::vector<double>& data) const;
  Eigen::Matrix4d ConstructTransformFromParameters(double x,
                                                   double y,
                                                   double z,
                                                   double t) const;
  PointT ComputeResultSpread(const HypothesisPtr& result) const;
  HypothesisPtr RefineResult(const HypothesisPtr& result) const;

  size_t algorithm_ = 0;
  double grid_size_ = 0.1;
  size_t icp_refinement_ = 0;
};

} // namespace map_matcher
