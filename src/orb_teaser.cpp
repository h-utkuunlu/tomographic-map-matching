#include <algorithm>
#include <iterator>
#include <memory>
#include <opencv2/core/types.hpp>
#include <teaser/registration.h>
#include <tomographic_map_matching/orb_teaser.hpp>

namespace map_matcher {

ORBTEASER::ORBTEASER()
  : TomographicMatcher()
{
}

ORBTEASER::ORBTEASER(const json& parameters)
  : TomographicMatcher(parameters)
{
  UpdateParameters(parameters);
}

void
ORBTEASER::GetParameters(json& output) const
{
  TomographicMatcher::GetParameters(output);
  output["teaser_num_correspondences_max"] = teaser_num_correspondences_max_;
  output["teaser_noise_bound"] = teaser_noise_bound_;
  output["teaser_verbose"] = teaser_verbose_;
  output["teaser_3d"] = teaser_3d_;
}

void
ORBTEASER::UpdateParameters(const json& input)
{
  UpdateSingleParameter(
    input, "teaser_num_correspondences_max", teaser_num_correspondences_max_);
  UpdateSingleParameter(input, "teaser_noise_bound", teaser_noise_bound_);
  UpdateSingleParameter(input, "teaser_verbose", teaser_verbose_);
  UpdateSingleParameter(input, "teaser_3d", teaser_3d_);
}

HypothesisPtr
ORBTEASER::RegisterPointCloudMaps(const PointCloud::Ptr source,
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

  HypothesisPtr result;

  if (teaser_3d_) {
    // Method 2: Compare features across all slices (in 3D)
    result = RunTeaserWith3DMatches(source_slice, target_slice);
  } else {
    // Method 1: Similar to correlations before, using TEASER
    std::vector<HypothesisPtr> correlation_results =
      CorrelateSlices(source_slice, target_slice);
    result = correlation_results[0];
  }
  stats["t_pose_estimation"] = CalculateTimeSince(indiv);

  // TODO: Verify if this makes sense
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
ORBTEASER::CorrelateSlices(const std::vector<SlicePtr>& source_features,
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

      HypothesisPtr hypothesis =
        RegisterForGivenInterval(source_features, target_features, indices);
      correlated_results.push_back(hypothesis);
    }

    // Update indices
    count++;
    if (source_index != 0)
      --source_index;
    else
      ++target_index;
  }

  // spdlog::info("Number of correlations num_correlation: {}", count);

  // Providing a lambda sorting function to deal with the use of smart
  // pointers. Otherwise sorted value is not exactly accurate
  std::sort(correlated_results.rbegin(),
            correlated_results.rend(),
            [](HypothesisPtr val1, HypothesisPtr val2) { return *val1 < *val2; });
  return correlated_results;
}

HypothesisPtr
ORBTEASER::RegisterForGivenInterval(const std::vector<SlicePtr>& source,
                                    const std::vector<SlicePtr>& target,
                                    HeightIndices indices) const
{
  if (indices.m2_max - indices.m2_min != indices.m1_max - indices.m1_min) {
    spdlog::critical("Different number of slices are sent for calculation");
    throw std::runtime_error("Different number of slices");
  }

  size_t window =
    std::min(indices.m2_max - indices.m2_min, indices.m1_max - indices.m1_min);

  // Aggregate matching features from all slices in the given range
  PointCloud::Ptr target_points(new PointCloud()), source_points(new PointCloud());
  std::vector<float> distances;

  for (size_t i = 0; i < window; ++i) {
    // Extract correct slice & assign the weak ptrs
    size_t m1_idx = indices.m1_min + i, m2_idx = indices.m2_min + i;
    const Slice &target_slice = *target[m1_idx], source_slice = *source[m2_idx];

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
    MatchingResultPtr matched_keypoints;
    if (gms_matching_) {
      matched_keypoints = MatchKeyPointsGMS(source_slice, target_slice);
    } else {
      matched_keypoints = MatchKeyPoints(source_slice, target_slice);
    }

    // Convert to image coordinates
    std::vector<cv::Point2f> points1img, points2img;
    cv::KeyPoint::convert(matched_keypoints->target_keypoints, points1img);
    cv::KeyPoint::convert(matched_keypoints->source_keypoints, points2img);

    // Convert to real coordinates (3D)
    PointCloud::Ptr points1 = img2real(
                      points1img, target_slice.slice_bounds, target_slice.height),
                    points2 = img2real(
                      points2img, source_slice.slice_bounds, source_slice.height);

    // Append to collective cloud
    *target_points += *points1;
    *source_points += *points2;
    distances.insert(distances.end(),
                     matched_keypoints->distances.begin(),
                     matched_keypoints->distances.end());
  }

  // Teaser++ registration on top N matches
  SelectTopNMatches(source_points, target_points, distances);
  spdlog::debug("Number of correspondences: {}", target_points->size());

  if (target_points->size() < 5) {
    spdlog::debug("Not enough correspondences");
    return HypothesisPtr(new Hypothesis());
  }

  std::shared_ptr<teaser::RobustRegistrationSolver> solver =
    RegisterPointsWithTeaser(target_points, source_points);

  // Construct solution
  HypothesisPtr result =
    ConstructSolutionFromSolverState(solver, source_points, target_points);

  return result;
}

std::shared_ptr<teaser::RobustRegistrationSolver>
ORBTEASER::RegisterPointsWithTeaser(const PointCloud::Ptr pcd1,
                                    const PointCloud::Ptr pcd2) const
{
  // Convert to Eigen
  size_t N = pcd1->size();
  Eigen::Matrix<double, 3, Eigen::Dynamic> pcd1eig(3, N), pcd2eig(3, N);
  for (size_t i = 0; i < N; ++i) {
    const PointT &pt1 = pcd1->points[i], pt2 = pcd2->points[i];
    pcd1eig.col(i) << pt1.x, pt1.y, pt1.z;
    pcd2eig.col(i) << pt2.x, pt2.y, pt2.z;
  }

  // Prepare solver based on system parameters
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = teaser_noise_bound_;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
    teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = 0.005;
  params.inlier_selection_mode =
    teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;

  std::unique_ptr<teaser::RobustRegistrationSolver> solver(
    new teaser::RobustRegistrationSolver(params));

  // Disable verbose output to stdout
  if (!teaser_verbose_)
    std::cout.setstate(std::ios_base::failbit);
  solver->solve(pcd2eig, pcd1eig);
  std::cout.clear();

  return solver;
}

HypothesisPtr
ORBTEASER::RunTeaserWith3DMatches(const std::vector<SlicePtr>& source_features,
                                  const std::vector<SlicePtr>& target_features) const
{
  // Extract all matches, slice by slice, in parallel
  PointCloud::Ptr target_points(new PointCloud()), source_points(new PointCloud());
  std::vector<float> distances;

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (size_t j = 0; j < source_features.size(); ++j) {
    for (size_t i = 0; i < target_features.size(); ++i) {
      const Slice &slice_target = *target_features[i],
                  slice_source = *source_features[j];

      MatchingResultPtr matches = MatchKeyPoints(slice_source, slice_target);

      std::vector<cv::Point2f> points1img, points2img;
      cv::KeyPoint::convert(matches->target_keypoints, points1img);
      cv::KeyPoint::convert(matches->source_keypoints, points2img);

      // Convert to real coordinates (3D)
      PointCloud::Ptr points1 = img2real(
                        points1img, slice_target.slice_bounds, slice_target.height),
                      points2 = img2real(
                        points2img, slice_source.slice_bounds, slice_source.height);

#pragma omp critical
      {
        *target_points += *points1;
        *source_points += *points2;
        distances.insert(distances.end(),
                         std::make_move_iterator(matches->distances.begin()),
                         std::make_move_iterator(matches->distances.end()));
      }
    }
  }

  // Retain only the top N, if larger than the teaser_num_correspondences_max
  SelectTopNMatches(source_points, target_points, distances);
  spdlog::debug("Number of correspondences: {}", target_points->size());
  if (target_points->size() < 5) {
    spdlog::debug("Not enough correspondences");
    return HypothesisPtr(new Hypothesis());
  }

  // Register
  std::shared_ptr<teaser::RobustRegistrationSolver> solver =
    RegisterPointsWithTeaser(target_points, source_points);

  // Process result from state
  HypothesisPtr result =
    ConstructSolutionFromSolverState(solver, source_points, target_points);

  return result;
}

void
ORBTEASER::SelectTopNMatches(PointCloud::Ptr& source_points,
                             PointCloud::Ptr& target_points,
                             const std::vector<float>& distances) const
{
  // Return as is if there are less matches than maximum
  if (distances.size() <= teaser_num_correspondences_max_)
    return;

  // Sort by indices
  std::vector<size_t> indices(distances.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Shortest distance (best match) at the front
  std::stable_sort(indices.begin(), indices.end(), [&distances](size_t i1, size_t i2) {
    return distances[i1] < distances[i2];
  });

  // Collate
  PointCloud::Ptr target_topN(new PointCloud()), source_topN(new PointCloud());
  for (size_t i = 0; i < teaser_num_correspondences_max_; ++i) {
    target_topN->push_back(target_points->points[indices[i]]);
    source_topN->push_back(source_points->points[indices[i]]);
  }

  target_points = target_topN;
  source_points = source_topN;
}

HypothesisPtr
ORBTEASER::ConstructSolutionFromSolverState(
  const std::shared_ptr<teaser::RobustRegistrationSolver>& solver,
  const PointCloud::Ptr& source_points,
  const PointCloud::Ptr& target_points) const
{
  HypothesisPtr result(new Hypothesis());
  result->inlier_points_1 = PointCloud::Ptr(new PointCloud());
  result->inlier_points_2 = PointCloud::Ptr(new PointCloud());

  std::vector<int> inlier_mask = solver->getInlierMaxClique();
  for (const auto& idx : inlier_mask) {
    const auto &pt1 = target_points->points[idx], &pt2 = source_points->points[idx];
    result->inlier_points_1->push_back(pt1);
    result->inlier_points_2->push_back(pt2);
  }

  result->n_inliers = result->inlier_points_1->size();

  // Decompose pose into parameters
  teaser::RegistrationSolution solution = solver->getSolution();
  result->x = solution.translation.x();
  result->y = solution.translation.y();
  result->z = solution.translation.z();

  Eigen::Matrix4d solution_mat = Eigen::Matrix4d::Identity();
  solution_mat.topLeftCorner(3, 3) = solution.rotation;
  solution_mat.topRightCorner(3, 1) = solution.translation;
  result->pose = solution_mat;

  Eigen::Vector3d eulAng = solution.rotation.eulerAngles(2, 1, 0);

  // Identify if the rotation axis is pointing downwards. In that case, the
  // rotation will be pi rad apart
  Eigen::AngleAxisd angle_axis(solution.rotation);
  if (angle_axis.axis()(2) < 0.0)
    result->theta = eulAng(0) - M_PI;
  else
    result->theta = eulAng(0);

  return result;
}

} // namespace map_matcher
