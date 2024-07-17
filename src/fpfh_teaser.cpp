#include <pcl/common/transforms.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <spdlog/spdlog.h>
#include <tomographic_map_matching/fpfh_teaser.hpp>

namespace map_matcher {

void to_json(json &j, const FPFHTEASERParameters &p) {
  to_json(j, static_cast<Parameters>(p));
  j["normal_radius"] = p.normal_radius;
  j["descriptor_radius"] = p.descriptor_radius;
  j["response_method"] = p.response_method;
  j["keypoint_radius"] = p.keypoint_radius;
  j["corner_threshold"] = p.corner_threshold;
  j["teaser_noise_bound"] = p.teaser_noise_bound;
  j["teaser_verbose"] = p.teaser_verbose;
}

void from_json(const json &j, FPFHTEASERParameters &p) {
  Parameters p_base;
  from_json(j, p_base);
  p = FPFHTEASERParameters(p_base);
  if (j.contains("normal_radius"))
    j.at("normal_radius").get_to(p.normal_radius);

  if (j.contains("descriptor_radius"))
    j.at("descriptor_radius").get_to(p.descriptor_radius);

  if (j.contains("keypoint_radius"))
    j.at("keypoint_radius").get_to(p.keypoint_radius);

  if (j.contains("response_method"))
    j.at("response_method").get_to(p.response_method);

  if (j.contains("corner_threshold"))
    j.at("corner_threshold").get_to(p.corner_threshold);

  if (j.contains("teaser_noise_bound"))
    j.at("teaser_noise_bound").get_to(p.teaser_noise_bound);

  if (j.contains("teaser_verbose"))
    j.at("teaser_verbose").get_to(p.teaser_verbose);
}

FPFHTEASER::FPFHTEASER() : MapMatcherBase() {}

FPFHTEASER::FPFHTEASER(FPFHTEASERParameters parameters)
    : MapMatcherBase(static_cast<Parameters>(parameters)),
      parameters_(parameters) {
  if (parameters_.response_method < 1 or parameters_.response_method > 5) {
    spdlog::warn("Corner response method must be in the range 1-5 (Harris, "
                 "Noble, Lowe, Tomasi, Curvature). Defaulting to Harris");
    parameters_.response_method = 1;
  }
}

json FPFHTEASER::GetParameters() const {
  json retval = parameters_;
  return retval;
}

void FPFHTEASER::SetParameters(const json &parameters) {
  parameters_ = parameters.template get<FPFHTEASERParameters>();

  if (parameters_.response_method < 1 or parameters_.response_method > 5) {
    spdlog::warn("Corner response method must be in the range 1-5 (Harris, "
                 "Noble, Lowe, Tomasi, Curvature). Defaulting to Harris");
    parameters_.response_method = 1;
  }
}

void FPFHTEASER::DetectAndDescribeKeypoints(const PointCloud::Ptr input,
                                            PointCloud::Ptr keypoints,
                                            FeatureCloud::Ptr features) const {

  spdlog::debug("PCD size: {}", input->size());

  std::chrono::steady_clock::time_point timer;
  timer = std::chrono::steady_clock::now();

  // Normals are needed for both keypoints and the FPFH
  NormalCloud::Ptr normals(new NormalCloud);
  pcl::NormalEstimationOMP<PointT, NormalT> normal_estimator;
  normal_estimator.setRadiusSearch(parameters_.normal_radius);
  normal_estimator.setInputCloud(input);
  normal_estimator.compute(*normals);

  spdlog::debug("Normal estimation took {} s", CalculateTimeSince(timer));
  timer = std::chrono::steady_clock::now();

  // Keypoints
  KeypointCloud::Ptr keypoints_with_response(new KeypointCloud);
  KeypointDetector detector;
  auto response_method = static_cast<KeypointDetector::ResponseMethod>(
      parameters_.response_method);
  detector.setMethod(response_method);
  detector.setRadius(parameters_.keypoint_radius);
  detector.setThreshold(parameters_.corner_threshold);
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
  descriptor.setRadiusSearch(parameters_.descriptor_radius);
  descriptor.setInputCloud(keypoints);
  descriptor.setSearchSurface(input);
  descriptor.setInputNormals(normals);
  descriptor.setSearchMethod(normal_estimator.getSearchMethod());
  descriptor.compute(*features);

  spdlog::debug("Feature computation took {} s", CalculateTimeSince(timer));
}

void FPFHTEASER::ExtractInlierKeypoints(
    const PointCloud::Ptr map1_pcd, const PointCloud::Ptr map2_pcd,
    const pcl::CorrespondencesPtr correspondences, PointCloud::Ptr map1_inliers,
    PointCloud::Ptr map2_inliers) const {

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

HypothesisPtr FPFHTEASER::RegisterPointCloudMaps(const PointCloud::Ptr map1_pcd,
                                                 const PointCloud::Ptr map2_pcd,
                                                 json &stats) const {

  if (map1_pcd->size() == 0 or map2_pcd->size() == 0) {
    spdlog::critical("Pointcloud(s) are empty. Aborting");
    return HypothesisPtr(new Hypothesis());
  }

  // Timing
  std::chrono::steady_clock::time_point total, indiv;
  total = std::chrono::steady_clock::now();
  indiv = std::chrono::steady_clock::now();

  // Compute keypoints and features
  PointCloud::Ptr map1_keypoints(new PointCloud),
      map2_keypoints(new PointCloud);
  FeatureCloud::Ptr map1_features(new FeatureCloud),
      map2_features(new FeatureCloud);

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
  ExtractInlierKeypoints(map1_keypoints, map2_keypoints,
                         correspondences_one_to_one, map1_inliers,
                         map2_inliers);

  // Registration with TEASER++
  HypothesisPtr result(new Hypothesis());

  {
    // Convert to Eigen
    size_t N = map1_inliers->size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> pcd1eig(3, N), pcd2eig(3, N);
    for (size_t i = 0; i < N; ++i) {
      const PointT &pt1 = map1_inliers->points[i],
                   pt2 = map2_inliers->points[i];
      pcd1eig.col(i) << pt1.x, pt1.y, pt1.z;
      pcd2eig.col(i) << pt2.x, pt2.y, pt2.z;
    }

    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = parameters_.teaser_noise_bound;
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
    if (!parameters_.teaser_verbose)
      std::cout.setstate(std::ios_base::failbit);
    teaser::RegistrationSolution solution = solver->solve(pcd2eig, pcd1eig);
    std::cout.clear();

    // Construct solution
    result->inlier_points_1 = PointCloud::Ptr(new PointCloud());
    result->inlier_points_2 = PointCloud::Ptr(new PointCloud());

    std::vector<int> inlier_mask = solver->getInlierMaxClique();
    for (const auto &idx : inlier_mask) {
      const auto &pt1 = map1_inliers->points[idx],
                 &pt2 = map2_inliers->points[idx];
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

void FPFHTEASER::VisualizeKeypoints(const PointCloud::Ptr points,
                                    const PointCloud::Ptr keypoints) const {

  pcl::visualization::PCLVisualizer viewer("Keypoints");
  PointCloudColor points_color(points, 155, 0, 0),
      keypoints_color(keypoints, 0, 155, 0);

  viewer.addPointCloud(points, points_color, "points");
  viewer.addPointCloud(keypoints, keypoints_color, "keypoints");

  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");

  viewer.spin();
}

} // namespace map_matcher
