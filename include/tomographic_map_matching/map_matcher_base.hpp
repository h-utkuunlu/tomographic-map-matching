#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace map_matcher {

using json = nlohmann::json;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT>
    PointCloudColor;
typedef pcl::visualization::PointCloudColorHandlerGenericField<PointT>
    PointCloudColorGF;

struct Parameters {

  Parameters() = default;
  Parameters(const Parameters &) = default;

  size_t algorithm = 0;
  double grid_size = 0.1;
  double slice_z_height = 0.1;
  double minimum_z_overlap_percentage = 0.0;
  size_t icp_refinement = 0;
  bool cross_match = false;
  bool median_filter = false;
  bool approximate_neighbors = false;

  bool gms_matching = false;
  double gms_threshold_factor = 0.5;

  size_t lsh_num_tables = 12;
  size_t lsh_key_size = 20;
  size_t lsh_multiprobe_level = 2;

  double orb_scale_factor = 1.2;
  int orb_num_features = 1000;
  int orb_n_levels = 8;
  int orb_edge_threshold = 31;
  int orb_first_level = 0;
  int orb_wta_k = 2;
  int orb_patch_size = 31;
  int orb_fast_threshold = 20;
};

void to_json(json &j, const Parameters &p);
void from_json(const json &j, Parameters &p);

struct CartesianBounds {
  CartesianBounds() {
    const double lowest = std::numeric_limits<double>::lowest(),
                 largest = std::numeric_limits<double>::max();
    upper = PointT(lowest, lowest, lowest);
    lower = PointT(largest, largest, largest);
  }
  CartesianBounds(PointT upper_input, PointT lower_input) {
    upper = upper_input;
    lower = lower_input;
  }
  PointT upper, lower;
};

struct Slice {
  Slice() {
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

struct SliceTransform {
  SliceTransform() {
    inliers = std::vector<std::pair<cv::Point2f, cv::Point2f>>();
    x = 0.0;
    y = 0.0;
    z = 0.0;
    theta = 0.0;
    pose = Eigen::Matrix4d::Identity();
  }

  std::shared_ptr<Slice> slice1, slice2;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> inliers;
  double x, y, z, theta;
  Eigen::Matrix4d pose;
};
typedef std::shared_ptr<SliceTransform> SliceTransformPtr;

struct HeightIndices {
  size_t m1_min, m1_max, m2_min, m2_max;
};

struct MatchingResult {
  MatchingResult() {
    map1_keypoints = std::vector<cv::KeyPoint>();
    map2_keypoints = std::vector<cv::KeyPoint>();
    distances = std::vector<float>();
  }

  std::vector<cv::KeyPoint> map1_keypoints, map2_keypoints;
  std::vector<float> distances;
};
typedef std::shared_ptr<MatchingResult> MatchingResultPtr;

struct Hypothesis {
  Hypothesis() {
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

  Hypothesis(const Hypothesis &other) {
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

  bool operator<(const Hypothesis &rhs) { return n_inliers < rhs.n_inliers; }
};
typedef std::shared_ptr<Hypothesis> HypothesisPtr;

class MapMatcherBase {
public:
  virtual HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                                               const PointCloud::Ptr pcd2,
                                               json &stats) const = 0;
  virtual json GetParameters() const = 0;
  virtual std::string GetName() const = 0;

  void VisualizeHypothesisSlices(const HypothesisPtr hypothesis) const;
  void VisualizeHypothesis(const PointCloud::Ptr &map1_pcd,
                           const PointCloud::Ptr &map2_pcd,
                           const HypothesisPtr &result) const;

protected:
  MapMatcherBase();
  MapMatcherBase(Parameters parameters);
  Parameters parameters_;

  size_t GetPeakRSS() const;
  double
  CalculateTimeSince(const std::chrono::steady_clock::time_point &start) const;
  CartesianBounds CalculateBounds(const PointCloud::Ptr &pcd) const;
  PointCloud::Ptr ExtractSlice(const PointCloud::Ptr &pcd, double height) const;
  std::vector<SlicePtr> ComputeSliceImages(const PointCloud::Ptr &map) const;
  std::vector<SlicePtr> &
  ComputeSliceFeatures(std::vector<SlicePtr> &image_slices) const;
  PointT CalculateXYZSpread(const PointCloud::Ptr &pcd) const;
  double EstimateStdDev(const std::vector<double> &data) const;
  void ConvertPCDSliceToImage(const PointCloud::Ptr &pcd_slice,
                              Slice &slice) const;
  std::vector<cv::Point2f> img2real(const std::vector<cv::Point2f> &pts,
                                    const CartesianBounds &mapBounds) const;
  PointCloud::Ptr img2real(const std::vector<cv::Point2f> &pts,
                           const CartesianBounds &mapBounds,
                           double z_height) const;
  MatchingResultPtr MatchKeyPoints(const Slice &slice1,
                                   const Slice &slice2) const;
  MatchingResultPtr MatchKeyPointsGMS(const Slice &slice1,
                                      const Slice &slice2) const;
  Eigen::Matrix4d ConstructTransformFromParameters(double x, double y, double z,
                                                   double t) const;
  void VisualizeSlice(const SlicePtr slice, std::string window_name) const;
  PointT ComputeResultSpread(const HypothesisPtr &result) const;
  HypothesisPtr RefineResult(const HypothesisPtr &result) const;
};

} // namespace map_matcher
