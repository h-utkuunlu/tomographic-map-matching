#pragma once

#include "tomographic_map_matching/consensus.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <map_matcher_interfaces/msg/slice_map.hpp>

// Conversions
namespace map_matcher_ros {

void
ConvertToROS(const std::vector<map_matcher::SlicePtr>& in,
             map_matcher_interfaces::msg::SliceMap& out)
{
  out.sliced_map.resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    auto& out_slice = out.sliced_map[i];
    const auto& in_slice = in[i];

    out_slice.height = in_slice->height;
    out_slice.is_descriptive = in_slice->is_descriptive;

    out_slice.slice_bounds.upper.x = in_slice->slice_bounds.upper.x;
    out_slice.slice_bounds.upper.y = in_slice->slice_bounds.upper.y;
    out_slice.slice_bounds.upper.z = in_slice->slice_bounds.upper.z;
    out_slice.slice_bounds.lower.x = in_slice->slice_bounds.lower.x;
    out_slice.slice_bounds.lower.y = in_slice->slice_bounds.lower.y;
    out_slice.slice_bounds.lower.z = in_slice->slice_bounds.lower.z;

    const size_t N_kp = in_slice->kp.size();
    out_slice.keypoints.resize(N_kp);
    for (size_t j = 0; j < N_kp; ++j) {
      out_slice.keypoints[j].x = in_slice->kp[j].pt.x;
      out_slice.keypoints[j].y = in_slice->kp[j].pt.y;
    }

    cv_bridge::CvImage bridge(
      std_msgs::msg::Header(), sensor_msgs::image_encodings::MONO8, in_slice->desc);
    out_slice.descriptors = *(bridge.toImageMsg());
  }
}

void
ConvertFromROS(const map_matcher_interfaces::msg::SliceMap& in,
               std::vector<map_matcher::SlicePtr>& out)
{
  const size_t N = in.sliced_map.size();
  out.resize(N);

  for (size_t i = 0; i < N; ++i) {
    out[i] = std::make_shared<map_matcher::Slice>();
    auto& out_slice = out[i];
    const auto& in_slice = in.sliced_map[i];

    out_slice->height = in_slice.height;
    out_slice->is_descriptive = in_slice.is_descriptive;

    out_slice->slice_bounds.upper.x = in_slice.slice_bounds.upper.x;
    out_slice->slice_bounds.upper.y = in_slice.slice_bounds.upper.y;
    out_slice->slice_bounds.upper.z = in_slice.slice_bounds.upper.z;
    out_slice->slice_bounds.lower.x = in_slice.slice_bounds.lower.x;
    out_slice->slice_bounds.lower.y = in_slice.slice_bounds.lower.y;
    out_slice->slice_bounds.lower.z = in_slice.slice_bounds.lower.z;

    const size_t N_kp = in_slice.keypoints.size();
    out_slice->kp.resize(N_kp);
    for (size_t j = 0; j < N_kp; ++j) {
      out_slice->kp[j].pt.x = in_slice.keypoints[j].x;
      out_slice->kp[j].pt.y = in_slice.keypoints[j].y;
    }

    auto bridge = cv_bridge::toCvCopy(in_slice.descriptors, "mono8");
    out_slice->desc = bridge->image;

    // FIXME: Hardcoding the matcher parameters
    out_slice->matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    out_slice->matcher->add(out_slice->desc);
    out_slice->matcher->train();
  }
}

} // namespace map_matcher_ros
