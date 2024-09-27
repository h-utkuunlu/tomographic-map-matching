#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/common/transforms.h>
#include <tomographic_map_matching/consensus.hpp>
#include <tomographic_map_matching/estimate_rigid_2d.hpp>

namespace map_matcher {

Consensus::Consensus()
  : TomographicMatcher()
{
}

Consensus::Consensus(const json& parameters)
  : TomographicMatcher(parameters)
{
  UpdateParameters(parameters);
}

void
Consensus::UpdateParameters(const json& input)
{
  UpdateSingleParameter(input, "consensus_ransac_factor", consensus_ransac_factor_);
  UpdateSingleParameter(input, "consensus_use_rigid", consensus_use_rigid_);
}

void
Consensus::GetParameters(json& output) const
{
  TomographicMatcher::GetParameters(output);
  output["consensus_ransac_factor"] = consensus_ransac_factor_;
  output["consensus_use_rigid"] = consensus_use_rigid_;
}

HypothesisPtr
Consensus::RegisterPointCloudMaps(const PointCloud::Ptr source,
                                  const PointCloud::Ptr target,
                                  json& stats) const
{

  if (target->size() == 0 or source->size() == 0) {
    spdlog::critical("Pointcloud(s) are empty. Aborting");
    return HypothesisPtr(new Hypothesis());
  }

  // Timing
  std::chrono::steady_clock::time_point total, indiv;
  total = std::chrono::steady_clock::now();
  indiv = std::chrono::steady_clock::now();

  // Calculate & store all possible binary images (determined by grid size)
  std::vector<SlicePtr> target_slice = ComputeSliceImages(target),
                        source_slice = ComputeSliceImages(source);

  stats["t_image_generation"] = CalculateTimeSince(indiv);
  stats["target_num_slices"] = target_slice.size();
  stats["source_num_slices"] = source_slice.size();

  // VisualizeImageSlices(target_image);
  // VisualizeImageSlices(source_image);
  indiv = std::chrono::steady_clock::now();

  // Convert binary images to feature slices
  ComputeSliceFeatures(target_slice);
  ComputeSliceFeatures(source_slice);

  // Calculate number of features in each map
  size_t target_nfeat = 0, source_nfeat = 0;
  for (const auto& slice : target_slice)
    target_nfeat += slice->kp.size();
  for (const auto& slice : source_slice)
    source_nfeat += slice->kp.size();

  stats["t_feature_extraction"] = CalculateTimeSince(indiv);
  stats["target_num_features"] = target_nfeat;
  stats["source_num_features"] = source_nfeat;
  indiv = std::chrono::steady_clock::now();

  std::vector<HypothesisPtr> slice_correlations =
    CorrelateSlices(source_slice, target_slice);

  HypothesisPtr result = slice_correlations[0];

  stats["t_pose_estimation"] = CalculateTimeSince(indiv);
  stats["num_hypothesis_inliers"] = result->n_inliers;

  if (result->n_inliers == 0) {
    spdlog::warn("Pose cannot be calculated");
  } else {
    // Spread analysis
    indiv = std::chrono::steady_clock::now();
    PointT spread = ComputeResultSpread(result);
    stats["t_spread_analysis"] = CalculateTimeSince(indiv);
    stats["spread_ax1"] = spread.x;
    stats["spread_ax2"] = spread.y;
    stats["spread_axz"] = spread.z;
    stats["num_feature_inliers"] = result->inlier_points_1->size();
  }
  stats["t_total"] = CalculateTimeSince(total);

  // Measure memory use
  stats["mem_cpu"] = GetPeakRSS();

  return result;
}

std::vector<HypothesisPtr>
Consensus::CorrelateSlices(const std::vector<SlicePtr>& source_features,
                           const std::vector<SlicePtr>& target_features) const
{
  // Number of possibilities (unless restricted) for slice pairings is n1 + n2 -
  // 1 Starting from bottom slice of m2 and top slice of m1 only, all the way to
  // the other way around. Manipulate index ranges
  size_t target_index = 0, source_index = source_features.size() - 1;
  const size_t target_size = target_features.size(),
               source_size = source_features.size();

  // Only consider overlaps of particular percentage
  const size_t minimum_overlap = static_cast<size_t>(
    std::round(minimum_z_overlap_percentage_ *
               static_cast<double>(std::min(target_size, source_size))));

  std::vector<HypothesisPtr> correlated_results;

  size_t count = 0;
  while (!(target_index == target_size && source_index == 0)) {
    // Height is determined by whichever has the smaller number of slices after
    // the index remaining between the two
    size_t height = std::min(target_size - target_index, source_size - source_index);

    if (height >= minimum_overlap) {
      HeightIndices indices{
        target_index, target_index + height, source_index, source_index + height
      };

      std::vector<SliceTransformPtr> results,
        results_interim = ComputeMapTf(source_features, target_features, indices);

      // Eliminate poses with zero inliers
      for (auto res : results_interim) {
        if (res->inliers.size())
          results.push_back(res);
      }

      HypothesisPtr agreed_result = VoteBetweenSlices(results);
      correlated_results.push_back(agreed_result);
    }

    // Update indices
    ++count;
    if (source_index != 0)
      --source_index;
    else
      ++target_index;
  }
  // spdlog::info("Num. correlations num_correlation: {}", count);

  // Providing a lambda sorting function to deal with the use of smart
  // pointers. Otherwise sorted value is not exactly accurate
  std::sort(correlated_results.rbegin(),
            correlated_results.rend(),
            [](HypothesisPtr val1, HypothesisPtr val2) { return *val1 < *val2; });
  return correlated_results;
}

std::vector<SliceTransformPtr>
Consensus::ComputeMapTf(const std::vector<SlicePtr>& source,
                        const std::vector<SlicePtr>& target,
                        HeightIndices indices) const
{
  if (indices.m2_max - indices.m2_min != indices.m1_max - indices.m1_min) {
    spdlog::critical("Different number of slices are sent for calculation");
    throw std::runtime_error("Different number of slices");
  }

  size_t window =
    std::min(indices.m2_max - indices.m2_min, indices.m1_max - indices.m1_min);
  std::vector<SliceTransformPtr> res(window);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < window; ++i) {
    // Initialize index
    res[i] = SliceTransformPtr(new SliceTransform());

    // Extract correct slice & assign the weak ptrs
    size_t m1_idx = indices.m1_min + i, m2_idx = indices.m2_min + i;
    const Slice &target_slice = *target[m1_idx], source_slice = *source[m2_idx];

    res[i]->target_slice = target[m1_idx];
    res[i]->source_slice = source[m2_idx];

    if (target_slice.kp.size() < 2 || source_slice.kp.size() < 2) {
      spdlog::debug(
        "Not enough keypoints in slices: m1_idx: {} kp: {} m2_idx: {} kp: {}",
        m1_idx,
        target_slice.kp.size(),
        m2_idx,
        source_slice.kp.size());
      continue;
    }

    // Extract matching keypoints
    MatchingResultPtr matched_keypoints = MatchKeyPoints(source_slice, target_slice);
    std::vector<cv::KeyPoint> kp1match = matched_keypoints->target_keypoints,
                              kp2match = matched_keypoints->source_keypoints;

    if (kp1match.size() <= 4) {
      spdlog::debug("Not enough matches. m1_idx: {} m2_idx: {} num_matches: {}",
                    m1_idx,
                    m2_idx,
                    kp1match.size());
      continue;
    }

    // Convert to image coordinates
    std::vector<cv::Point2f> points1img, points2img;
    cv::KeyPoint::convert(kp1match, points1img);
    cv::KeyPoint::convert(kp2match, points2img);

    // Convert to real coordinates
    std::vector<cv::Point2f> points1 = img2real(points1img, target_slice.slice_bounds);
    std::vector<cv::Point2f> points2 = img2real(points2img, source_slice.slice_bounds);

    cv::Mat inliers;

    // Coordinates used in estimation are in real coordinates. Making the RANSAC
    // threshold to be dependent on the grid size (resolution) of the maps,
    // instead of a fixed value, so that the threshold is to an extent uniform
    // across different resolution maps
    double ransacReprojThresh = grid_size_ * 3.0; // Default: 3.0
    size_t maxIters = 2000;                       // Default: 2000
    double confidence = 0.999;                    // Default: 0.99
    size_t refineIters = 10;                      // Default: 10

    cv::Mat tf;
    if (consensus_use_rigid_)
      tf = estimateRigid2D(points2,
                           points1,
                           inliers,
                           cv::RANSAC,
                           ransacReprojThresh,
                           maxIters,
                           confidence,
                           refineIters);
    else
      tf = cv::estimateAffinePartial2D(points2,
                                       points1,
                                       inliers,
                                       cv::RANSAC,
                                       ransacReprojThresh,
                                       maxIters,
                                       confidence,
                                       refineIters);

    // Extract inlier corresponding points
    std::vector<std::pair<cv::Point2f, cv::Point2f>> inliers_vec;
    size_t idx_count = 0;
    for (cv::MatIterator_<uchar> it = inliers.begin<uchar>();
         it != inliers.end<uchar>();
         ++it) {
      if (*it == 1) {
        inliers_vec.push_back(std::make_pair(points1[idx_count], points2[idx_count]));
      }
      ++idx_count;
    }

    if (inliers_vec.empty()) {
      spdlog::debug("Not enough inliers. m1_idx: {} m2_idx: {}", m1_idx, m2_idx);
      continue;
    }

    // Extract components
    double theta = std::atan2(tf.at<double>(1, 0), tf.at<double>(0, 0));
    double x = tf.at<double>(0, 2), y = tf.at<double>(1, 2);
    double scale = tf.at<double>(0, 0) / std::cos(theta);
    spdlog::debug("estimate*2D scale: {}", scale);

    // Height difference between matching slices is sufficient and should be
    // consistent between different maps since the grid size is the same
    double z = target[m1_idx]->height - source[m2_idx]->height;

    Eigen::Affine3d pose = Eigen::Affine3d::Identity();
    pose.translation() << x, y, z;
    pose.rotate(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));

    res[i]->inliers = inliers_vec;
    res[i]->x = x;
    res[i]->y = y;
    res[i]->z = z;
    res[i]->theta = theta;
    res[i]->pose = pose.matrix();
  }
  return res;
}

HypothesisPtr
Consensus::VoteBetweenSlices(const std::vector<SliceTransformPtr>& results) const
{
  std::vector<HypothesisPtr> voted_results(results.size());
  double dist_thresh = grid_size_ * consensus_ransac_factor_,
         tThresh = 0.015; // ~ 15 deg

  // Skip if there are no slices to vote for
  if (results.size() == 0) {
    return HypothesisPtr(new Hypothesis());
  }

  // Since the number of slices is small, check RANSAC over all slice hypotheses
  for (size_t i = 0; i < results.size(); ++i) {
    size_t n_inliers = 0;
    double xAvg = 0.0, yAvg = 0.0, tSinAvg = 0.0, tCosAvg = 0.0;
    std::vector<SliceTransformPtr> inlier_slices;

    // Collective update
    for (size_t j = 0; j < results.size(); ++j) {
      const SliceTransformPtr k = results[j];
      double dx = results[i]->x - k->x, dy = results[i]->y - k->y,
             dist = dx * dx + dy * dy;
      if (dist < dist_thresh * dist_thresh &&
          std::abs(std::cos(results[i]->theta) - std::cos(k->theta)) < tThresh) {
        ++n_inliers;
        xAvg += k->x;
        yAvg += k->y;
        tSinAvg += std::sin(k->theta);
        tCosAvg += std::cos(k->theta);
        inlier_slices.push_back(k);
      }
    }
    double n_inliers_d = static_cast<double>(n_inliers);

    xAvg /= n_inliers_d;
    yAvg /= n_inliers_d;
    tSinAvg /= n_inliers_d;
    tCosAvg /= n_inliers_d;
    double tAvg = std::atan2(tSinAvg, tCosAvg);

    std::vector<double> cur_hyp{ xAvg, yAvg, tAvg };

    voted_results[i] = HypothesisPtr(new Hypothesis);
    voted_results[i]->n_inliers = n_inliers;
    voted_results[i]->x = xAvg;
    voted_results[i]->y = yAvg;
    voted_results[i]->z = results[i]->z;
    voted_results[i]->theta = tAvg;
    // Move vector to avoid copying
    voted_results[i]->inlier_slices = std::move(inlier_slices);
    voted_results[i]->pose =
      ConstructTransformFromParameters(xAvg, yAvg, results[i]->z, tAvg);
  }

  std::sort(voted_results.rbegin(),
            voted_results.rend(),
            [](HypothesisPtr val1, HypothesisPtr val2) { return *val1 < *val2; });
  return voted_results[0];
}

} // namespace map_matcher
