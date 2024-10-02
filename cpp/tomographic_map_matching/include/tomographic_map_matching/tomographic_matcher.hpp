#pragma once

#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

struct MatchingResult
{
  MatchingResult()
  {
    target_keypoints = std::vector<cv::KeyPoint>();
    source_keypoints = std::vector<cv::KeyPoint>();
    distances = std::vector<float>();
  }

  std::vector<cv::KeyPoint> target_keypoints, source_keypoints;
  std::vector<float> distances;
};
typedef std::shared_ptr<MatchingResult> MatchingResultPtr;

class TomographicMatcher : public MapMatcherBase
{

public:
  void VisualizeHypothesisSlices(const HypothesisPtr hypothesis) const;
  virtual void UpdateParameters(const json& input) override;
  virtual void GetParameters(json& output) const override;
  void VisualizeSlice(const SlicePtr slice, std::string window_name) const;
  std::vector<SlicePtr> ComputeSliceImages(const PointCloud::Ptr& map) const;
  std::vector<SlicePtr>& ComputeSliceFeatures(
    std::vector<SlicePtr>& image_slices) const;

protected:
  TomographicMatcher();
  TomographicMatcher(const json parameters);

  PointCloud::Ptr ExtractSlice(const PointCloud::Ptr& pcd, double height) const;
  void ConvertPCDSliceToImage(const PointCloud::Ptr& pcd_slice, Slice& slice) const;
  std::vector<cv::Point2f> img2real(const std::vector<cv::Point2f>& pts,
                                    const CartesianBounds& mapBounds) const;
  PointCloud::Ptr img2real(const std::vector<cv::Point2f>& pts,
                           const CartesianBounds& mapBounds,
                           double z_height) const;

  MatchingResultPtr MatchKeyPoints(const Slice& source_slice,
                                   const Slice& target_slice) const;

  bool cross_match_ = false;
  bool approximate_neighbors_ = false;
  double minimum_z_overlap_percentage_ = 0.0;
  double slice_z_height_ = 0.1;
  bool median_filter_ = false;

  size_t lsh_num_tables_ = 12;
  size_t lsh_key_size_ = 20;
  size_t lsh_multiprobe_level_ = 2;

  double orb_scale_factor_ = 1.2;
  int orb_num_features_ = 1000;
  int orb_n_levels_ = 8;
  int orb_edge_threshold_ = 31;
  int orb_first_level_ = 0;
  int orb_wta_k_ = 2;
  int orb_patch_size_ = 31;
  int orb_fast_threshold_ = 20;
};
} // namespace map_matcher
