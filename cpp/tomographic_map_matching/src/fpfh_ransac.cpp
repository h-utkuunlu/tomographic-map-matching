#include <pcl/common/transforms.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <tomographic_map_matching/fpfh_ransac.hpp>

namespace map_matcher {

FPFHRANSAC::FPFHRANSAC()
  : FPFHBase()
{
}

FPFHRANSAC::FPFHRANSAC(const json& parameters)
  : FPFHBase(parameters)
{
  UpdateParameters(parameters);
}

void
FPFHRANSAC::GetParameters(json& output) const
{
  FPFHBase::GetParameters(output);
  output["ransac_inlier_threshold"] = ransac_inlier_threshold_;
  output["ransac_refine_model"] = ransac_refine_model_;
}

void
FPFHRANSAC::UpdateParameters(const json& input)
{
  UpdateSingleParameter(input, "ransac_inlier_threshold", ransac_inlier_threshold_);
  UpdateSingleParameter(input, "ransac_refine_model", ransac_refine_model_);
}

HypothesisPtr
FPFHRANSAC::RegisterPointCloudMaps(const PointCloud::Ptr source,
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

  // Compute keypoints and features
  PointCloud::Ptr target_keypoints(new PointCloud), source_keypoints(new PointCloud);
  FeatureCloud::Ptr target_features(new FeatureCloud),
    source_features(new FeatureCloud);

  DetectAndDescribeKeypoints(target, target_keypoints, target_features);
  DetectAndDescribeKeypoints(source, source_keypoints, source_features);

  stats["t_feature_extraction"] = CalculateTimeSince(indiv);
  stats["target_num_features"] = target_features->size();
  stats["source_num_features"] = source_features->size();

  spdlog::debug("Feature extraction took {} s",
                stats["t_feature_extraction"].template get<double>());
  indiv = std::chrono::steady_clock::now();

  // Matching & registration
  pcl::registration::CorrespondenceEstimation<FeatureT, FeatureT>
    correspondence_estimator;
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
  correspondence_estimator.setInputSource(source_features);
  correspondence_estimator.setInputTarget(target_features);
  correspondence_estimator.determineCorrespondences(*correspondences);
  spdlog::debug("Matching complete");

  // Limit to one-to-one matches
  pcl::CorrespondencesPtr correspondences_one_to_one(new pcl::Correspondences);
  pcl::registration::CorrespondenceRejectorOneToOne rejector_one_to_one;
  rejector_one_to_one.setInputCorrespondences(correspondences);
  rejector_one_to_one.getCorrespondences(*correspondences_one_to_one);
  spdlog::debug("One-to-one rejection complete");

  // Correspondance rejection with RANSAC
  pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> rejector_ransac;
  pcl::CorrespondencesPtr correspondences_inlier(new pcl::Correspondences);
  rejector_ransac.setInlierThreshold(ransac_inlier_threshold_);
  rejector_ransac.setRefineModel(ransac_refine_model_);

  rejector_ransac.setInputSource(source_keypoints);
  rejector_ransac.setInputTarget(target_keypoints);
  rejector_ransac.setInputCorrespondences(correspondences_one_to_one);
  rejector_ransac.getCorrespondences(*correspondences_inlier);
  Eigen::Matrix4f transform = rejector_ransac.getBestTransformation();

  stats["t_pose_estimation"] = CalculateTimeSince(indiv);
  spdlog::debug("Pose estimation completed in {}. Num. inliers: {}",
                stats["t_pose_estimation"].template get<double>(),
                correspondences_inlier->size());

  // Extract inliers
  PointCloud::Ptr target_inliers(new PointCloud), source_inliers(new PointCloud);
  ExtractInlierKeypoints(source_keypoints,
                         target_keypoints,
                         correspondences_inlier,
                         source_inliers,
                         target_inliers);

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
  result->pose =
    ConstructTransformFromParameters(result->x, result->y, result->z, result->theta);

  result->inlier_points_1 = target_inliers;
  result->inlier_points_2 = source_inliers;

  stats["t_total"] = CalculateTimeSince(total);

  // Measure memory use
  stats["mem_cpu"] = GetPeakRSS();

  // // Visualize features
  // VisualizeKeypoints(target, target_kp_coords);
  // VisualizeKeypoints(source, source_kp_coords);

  return result;
}

} // namespace map_matcher
