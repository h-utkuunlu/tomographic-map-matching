#include <pcl/common/transforms.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <tomographic_map_matching/fpfh_teaser.hpp>

namespace map_matcher {

FPFHTEASER::FPFHTEASER()
  : FPFHBase()
{
}

FPFHTEASER::FPFHTEASER(const json& parameters)
  : FPFHBase(parameters)
{
  UpdateParameters(parameters);
}

void
FPFHTEASER::GetParameters(json& output) const
{
  FPFHBase::GetParameters(output);
  output["teaser_noise_bound"] = teaser_noise_bound_;
  output["teaser_verbose"] = teaser_verbose_;
}

void
FPFHTEASER::UpdateParameters(const json& input)
{

  UpdateSingleParameter(input, "teaser_noise_bound", teaser_noise_bound_);
  UpdateSingleParameter(input, "teaser_verbose", teaser_verbose_);
}

HypothesisPtr
FPFHTEASER::RegisterPointCloudMaps(const PointCloud::Ptr map1_pcd,
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
  PointCloud::Ptr map1_keypoints(new PointCloud), map2_keypoints(new PointCloud);
  FeatureCloud::Ptr map1_features(new FeatureCloud), map2_features(new FeatureCloud);

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
  spdlog::debug("One-to-one rejection complete. Count: {}",
                correspondences_one_to_one->size());

  // Extract selected keypoints
  PointCloud::Ptr map1_inliers(new PointCloud), map2_inliers(new PointCloud);
  ExtractInlierKeypoints(map1_keypoints,
                         map2_keypoints,
                         correspondences_one_to_one,
                         map1_inliers,
                         map2_inliers);

  // Registration with TEASER++
  HypothesisPtr result(new Hypothesis());

  {
    // Convert to Eigen
    size_t N = map1_inliers->size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> pcd1eig(3, N), pcd2eig(3, N);
    for (size_t i = 0; i < N; ++i) {
      const PointT &pt1 = map1_inliers->points[i], pt2 = map2_inliers->points[i];
      pcd1eig.col(i) << pt1.x, pt1.y, pt1.z;
      pcd2eig.col(i) << pt2.x, pt2.y, pt2.z;
    }

    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = teaser_noise_bound_;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO;
    params.rotation_cost_threshold = 0.0002;
    params.inlier_selection_mode =
      teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;
    auto solver = std::make_unique<teaser::RobustRegistrationSolver>(params);

    // Disable verbose output to stdout
    if (!teaser_verbose_)
      std::cout.setstate(std::ios_base::failbit);
    teaser::RegistrationSolution solution = solver->solve(pcd2eig, pcd1eig);
    std::cout.clear();

    // Construct solution
    result->inlier_points_1 = PointCloud::Ptr(new PointCloud());
    result->inlier_points_2 = PointCloud::Ptr(new PointCloud());

    std::vector<int> inlier_mask = solver->getInlierMaxClique();
    for (const auto& idx : inlier_mask) {
      const auto &pt1 = map1_inliers->points[idx], &pt2 = map2_inliers->points[idx];
      result->inlier_points_1->push_back(pt1);
      result->inlier_points_2->push_back(pt2);
    }

    result->n_inliers = result->inlier_points_1->size();
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
  }

  stats["t_pose_estimation"] = CalculateTimeSince(indiv);
  stats["t_total"] = CalculateTimeSince(total);

  spdlog::debug("Pose estimation completed in {}. Num. inliers: {}",
                stats["t_pose_estimation"].template get<double>(),
                result->n_inliers);

  // Measure memory use
  stats["mem_cpu"] = GetPeakRSS();

  // // Visualize features
  // VisualizeKeypoints(map1_pcd, map1_kp_coords);
  // VisualizeKeypoints(map2_pcd, map2_kp_coords);

  return result;
}

} // namespace map_matcher
