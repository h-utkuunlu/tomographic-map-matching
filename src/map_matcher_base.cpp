#include <opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <spdlog/spdlog.h>
#include <sys/resource.h>
#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

void to_json(json &j, const Parameters &p) {
  j = json{{"algorithm", p.algorithm},
           {"grid_size", p.grid_size},
           {"slice_z_height", p.slice_z_height},
           {"minimum_z_overlap_percentage", p.minimum_z_overlap_percentage},
           {"icp_refinement", p.icp_refinement},
           {"cross_match", p.cross_match},
           {"median_filter", p.median_filter},
           {"approximate_neighbors", p.approximate_neighbors},
           {"gms_matching", p.gms_matching},
           {"gms_threshold_factor", p.gms_threshold_factor},
           {"lsh_num_tables", p.lsh_num_tables},
           {"lsh_key_size", p.lsh_key_size},
           {"lsh_multiprobe_level", p.lsh_multiprobe_level},
           {"orb_scale_factor", p.orb_scale_factor},
           {"orb_num_features", p.orb_num_features},
           {"orb_n_levels", p.orb_n_levels},
           {"orb_edge_threshold", p.orb_edge_threshold},
           {"orb_first_level", p.orb_first_level},
           {"orb_wta_k", p.orb_wta_k},
           {"orb_patch_size", p.orb_patch_size},
           {"orb_fast_threshold", p.orb_fast_threshold}};
}

void from_json(const json &j, Parameters &p) {
  j.at("algorithm").get_to(p.algorithm);
  j.at("grid_size").get_to(p.grid_size);
  j.at("slice_z_height").get_to(p.slice_z_height);
  j.at("minimum_z_overlap_percentage").get_to(p.minimum_z_overlap_percentage);
  j.at("icp_refinement").get_to(p.icp_refinement);
  j.at("cross_match").get_to(p.cross_match);
  j.at("median_filter").get_to(p.median_filter);
  j.at("approximate_neighbors").get_to(p.approximate_neighbors);
  j.at("gms_matching").get_to(p.gms_matching);
  j.at("gms_threshold_factor").get_to(p.gms_threshold_factor);
  j.at("lsh_num_tables").get_to(p.lsh_num_tables);
  j.at("lsh_key_size").get_to(p.lsh_key_size);
  j.at("lsh_multiprobe_level").get_to(p.lsh_multiprobe_level);
  j.at("orb_scale_factor").get_to(p.orb_scale_factor);
  j.at("orb_num_features").get_to(p.orb_num_features);
  j.at("orb_n_levels").get_to(p.orb_n_levels);
  j.at("orb_edge_threshold").get_to(p.orb_edge_threshold);
  j.at("orb_first_level").get_to(p.orb_first_level);
  j.at("orb_wta_k").get_to(p.orb_wta_k);
  j.at("orb_patch_size").get_to(p.orb_patch_size);
  j.at("orb_fast_threshold").get_to(p.orb_fast_threshold);
}

MapMatcherBase::MapMatcherBase() {}

MapMatcherBase::MapMatcherBase(Parameters parameters)
    : parameters_(parameters) {}

size_t MapMatcherBase::GetPeakRSS() const {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return (size_t)(rusage.ru_maxrss * 1024L);
}

double MapMatcherBase::CalculateTimeSince(
    const std::chrono::steady_clock::time_point &start) const {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
             .count() /
         1000000.0;
}

PointCloud::Ptr MapMatcherBase::ExtractSlice(const PointCloud::Ptr &pcd,
                                             double height) const {
  size_t N = pcd->size();
  double hmin = height - (parameters_.slice_z_height / 2.0),
         hmax = height + (parameters_.slice_z_height / 2.0);
  PointCloud::Ptr slice(new PointCloud());

  for (size_t i = 0; i < N; ++i) {
    const PointT &pt = (*pcd)[i];
    if (pt.z > hmin && pt.z < hmax) {
      slice->push_back(PointT(pt.x, pt.y, 0.0));
    }
  }

  return slice;
}

CartesianBounds
MapMatcherBase::CalculateBounds(const PointCloud::Ptr &pcd) const {
  auto low = std::numeric_limits<float>::lowest(),
       high = std::numeric_limits<float>::max();
  PointT upper(low, low, low), lower(high, high, high);

  for (const auto &pt : *pcd) {
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

void MapMatcherBase::ConvertPCDSliceToImage(const PointCloud::Ptr &pcd_slice,
                                            Slice &image_slice) const {
  size_t yPix = static_cast<size_t>(image_slice.binary_image.rows);
  // size_t xPix = static_cast<size_t>(image_slice.binary_image.cols);

  for (const auto &pt : *pcd_slice) {
    // Find coordinate of the point on the image
    size_t xIdx = std::round((pt.x - image_slice.slice_bounds.lower.x) /
                             parameters_.grid_size);
    size_t yIdx = std::round((pt.y - image_slice.slice_bounds.lower.y) /
                             parameters_.grid_size);

    // Flip y direction to match conventional image representation (y up, x
    // right)
    yIdx = yPix - yIdx - 1;
    image_slice.binary_image.at<uchar>(yIdx, xIdx) = 255;
  }

  // Median filter if selected
  if (parameters_.median_filter) {
    cv::medianBlur(image_slice.binary_image, image_slice.binary_image, 3);
  }
}

std::vector<cv::Point2f>
MapMatcherBase::img2real(const std::vector<cv::Point2f> &pts,
                         const CartesianBounds &mapBounds) const {
  // Needed to flip direction of y
  size_t yPix = std::round(
      (mapBounds.upper.y - mapBounds.lower.y) / parameters_.grid_size + 1.0);

  std::vector<cv::Point2f> converted(pts.size());
  size_t index = 0;

  for (const auto &pt : pts) {
    float x = pt.x * parameters_.grid_size + mapBounds.lower.x;
    float y = (yPix - pt.y - 1) * parameters_.grid_size + mapBounds.lower.y;
    converted[index++] = cv::Point2f(x, y);
  }

  return converted;
}

PointCloud::Ptr MapMatcherBase::img2real(const std::vector<cv::Point2f> &pts,
                                         const CartesianBounds &mapBounds,
                                         double z_height) const {
  // Needed to flip direction of y
  size_t yPix = std::round(
      (mapBounds.upper.y - mapBounds.lower.y) / parameters_.grid_size + 1.0);

  PointCloud::Ptr converted(new PointCloud());

  for (const auto &pt : pts) {
    float x = pt.x * parameters_.grid_size + mapBounds.lower.x;
    float y = (yPix - pt.y - 1) * parameters_.grid_size + mapBounds.lower.y;
    converted->push_back(PointT(x, y, z_height));
  }

  return converted;
}

std::vector<SlicePtr>
MapMatcherBase::ComputeSliceImages(const PointCloud::Ptr &map) const {
  CartesianBounds map_bounds = CalculateBounds(map);

  // Number of pixels in x- and y-directions are independent of individual point
  // cloud slices. Calculate it once
  size_t xPix = std::round(
      (map_bounds.upper.x - map_bounds.lower.x) / parameters_.grid_size + 1.0);
  size_t yPix = std::round(
      (map_bounds.upper.y - map_bounds.lower.y) / parameters_.grid_size + 1.0);

  spdlog::debug("Slice dimensions: ({}, {})", xPix, yPix);

  // Identify the number of slices that can be computed from this particular map
  double zmin = map_bounds.lower.z, zmax = map_bounds.upper.z;
  size_t maximum_index = static_cast<size_t>(
      std::round((zmax - zmin) / parameters_.grid_size) + 1);

  std::vector<SlicePtr> image_slices(maximum_index);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < maximum_index; ++i) {
    // Unified slice bounds for images to be the same size across different
    // layers
    double height = zmin + i * parameters_.grid_size;
    CartesianBounds slice_bounds(
        PointT(map_bounds.upper.x, map_bounds.upper.y, height),
        PointT(map_bounds.lower.x, map_bounds.lower.y, height));

    SlicePtr image_slice(new Slice());
    image_slice->height = height;
    image_slice->binary_image = cv::Mat(yPix, xPix, CV_8UC1, cv::Scalar(0));
    image_slice->slice_bounds = slice_bounds;
    image_slices[i] = image_slice;

    PointCloud::Ptr pcd_slice = ExtractSlice(map, height);

    // Descriptiveness determined by having enough points (more than linear sum
    // of image dimensions. Maybe a percentage of the overall size?) on the
    // pointcloud. If it does not, then it will be difficult to get useful
    // features from the image.
    image_slices[i]->is_descriptive = pcd_slice->size() >= (xPix + yPix);

    if (image_slices[i]->is_descriptive) {
      ConvertPCDSliceToImage(pcd_slice, *image_slices[i]);
    }
  }

  // Trim non-descriptive slices from the ends
  size_t start = 0, end = 0;
  bool start_trimmed = false, end_trimmed = false;

  for (size_t i = 0; i < maximum_index; ++i) {
    if (start_trimmed && end_trimmed)
      break;

    // From the front
    if (!(image_slices[i]->is_descriptive)) {
      if (!start_trimmed)
        ++start;
    } else
      start_trimmed = true;

    // From the end
    size_t j = maximum_index - i - 1;

    if (!(image_slices[j]->is_descriptive)) {
      if (!end_trimmed)
        ++end;
    } else
      end_trimmed = true;
  }

  spdlog::debug("Start trim: {} End trim: {}", start, end);

  std::vector<SlicePtr> image_slices_trimmed;
  image_slices_trimmed.insert(
      image_slices_trimmed.end(),
      std::make_move_iterator(image_slices.begin() + start),
      std::make_move_iterator(image_slices.end() - end));

  return image_slices_trimmed;
}

void MapMatcherBase::VisualizeHypothesisSlices(
    const HypothesisPtr hypothesis) const {
  size_t num_inlier_slices = hypothesis->inlier_slices.size(), current_idx = 0;

  if (num_inlier_slices == 0) {
    spdlog::warn(
        "There are no inlier slices for the hypothesis. Cannot visualize");
    return;
  }

  cv::namedWindow("slice1", cv::WINDOW_NORMAL);
  cv::namedWindow("slice2", cv::WINDOW_NORMAL);
  // cv::namedWindow("combined", cv::WINDOW_NORMAL);

  // Interactive visualization
  while (true) {
    spdlog::info("Pair {} / {}", current_idx + 1, num_inlier_slices);

    SliceTransformPtr &slice_pair = hypothesis->inlier_slices[current_idx];
    VisualizeSlice(slice_pair->slice1, "slice1");
    VisualizeSlice(slice_pair->slice2, "slice2");

    int key = cv::waitKey(0);

    // h key to decrement
    if (key == 104) {
      if (current_idx != 0)
        --current_idx;
    }

    // j key to increment
    if (key == 106) {
      if (current_idx < num_inlier_slices - 1)
        ++current_idx;
    }

    // escape / q key to exit
    if (key == 27 or key == 113) {
      break;
    }
  }

  cv::destroyAllWindows();
}

void MapMatcherBase::VisualizeSlice(const SlicePtr slice,
                                    std::string window_name) const {
  cv::Mat display_image;

  if (slice->is_descriptive) {
    // Single channel to 3 channels for color-coding
    cv::cvtColor(slice->binary_image, display_image, cv::COLOR_GRAY2BGR);
    cv::imshow(window_name, display_image);
  }
}

std::vector<SlicePtr> &MapMatcherBase::ComputeSliceFeatures(
    std::vector<SlicePtr> &image_slices) const {
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < image_slices.size(); ++i) {
    cv::Ptr<cv::ORB> detector = cv::ORB::create(
        parameters_.orb_num_features, parameters_.orb_scale_factor,
        parameters_.orb_n_levels, parameters_.orb_edge_threshold,
        parameters_.orb_first_level, parameters_.orb_wta_k,
        cv::ORB::HARRIS_SCORE, parameters_.orb_patch_size,
        parameters_.orb_fast_threshold);
    Slice &image_slice = *image_slices[i];
    size_t padding = detector->getEdgeThreshold();
    double padding_d = static_cast<double>(padding);

    // Do not waste time on slices that do not have enough points
    if (image_slice.is_descriptive) {
      cv::Mat padded_image;
      cv::copyMakeBorder(image_slice.binary_image, padded_image, padding,
                         padding, padding, padding, cv::BORDER_CONSTANT);

      detector->detectAndCompute(padded_image, cv::noArray(), image_slice.kp,
                                 image_slice.desc);

      // Subtract the padded coordinates
      for (auto &kp : image_slice.kp) {
        kp.pt.x -= padding_d;
        kp.pt.y -= padding_d;
      }

      // Matcher is embedded into the slice for faster querying
      if (parameters_.approximate_neighbors) {
        image_slice.matcher = cv::makePtr<cv::FlannBasedMatcher>(
            cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(
                parameters_.lsh_num_tables, parameters_.lsh_key_size,
                parameters_.lsh_multiprobe_level)));
      } else {
        if (parameters_.orb_wta_k == 2) {
          image_slice.matcher =
              cv::BFMatcher::create(cv::NORM_HAMMING, parameters_.cross_match);
        } else if (parameters_.orb_wta_k == 3 or parameters_.orb_wta_k == 4) {
          image_slice.matcher =
              cv::BFMatcher::create(cv::NORM_HAMMING2, parameters_.cross_match);
        } else {
          spdlog::critical("ORB WTA_K cannot be anything other than 2, 3, or "
                           "4. Set value: {}",
                           parameters_.orb_wta_k);
        }
      }
      image_slice.matcher->add(image_slice.desc);
      image_slice.matcher->train();
    }
  }
  return image_slices;
}

PointT MapMatcherBase::CalculateXYZSpread(const PointCloud::Ptr &pcd) const {
  double N = static_cast<double>(pcd->size());

  // Since we know the maps to be gravity aligned, we are only interested in the
  // spread along the xy plane. Using PCA to extract the major axes. For Z axis,
  // we only need the simple std. deviation estimate
  PointT mean(0.0, 0.0, 0.0);
  double z_sq = 0.0;
  Eigen::Matrix<double, 2, Eigen::Dynamic> X(2, pcd->size());

  // Accumulate matrix terms, means and z squared values
  size_t idx = 0;
  for (const PointT &pt : pcd->points) {
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

MatchingResultPtr MapMatcherBase::MatchKeyPoints(const Slice &slice1,
                                                 const Slice &slice2) const {
  MatchingResultPtr result(new MatchingResult());

  // Skip if not descriptive
  if (!(slice1.is_descriptive && slice2.is_descriptive))
    return result;

  if (parameters_.cross_match) {
    // Cross-matching
    std::vector<cv::DMatch> matches;
    slice2.matcher->match(slice1.desc, matches);

    for (const auto &match : matches) {
      result->map1_keypoints.push_back(slice1.kp[match.queryIdx]);
      result->map2_keypoints.push_back(slice2.kp[match.trainIdx]);
      result->distances.push_back(match.distance);
    }

  } else {
    // 2-way match with ratio test
    std::vector<std::vector<cv::DMatch>> knnMatches12, knnMatches21;
    slice2.matcher->knnMatch(slice1.desc, knnMatches12, 2);
    slice1.matcher->knnMatch(slice2.desc, knnMatches21, 2);

    const float ratioThresh = 0.7f;

    for (size_t i = 0; i < knnMatches12.size(); ++i) {
      if (knnMatches12[i].size() != 2)
        continue;

      if (knnMatches12[i][0].distance <
          ratioThresh * knnMatches12[i][1].distance) {
        result->map1_keypoints.push_back(
            slice1.kp[knnMatches12[i][0].queryIdx]);
        result->map2_keypoints.push_back(
            slice2.kp[knnMatches12[i][0].trainIdx]);
        result->distances.push_back(knnMatches12[i][0].distance);
      }
    }

    for (size_t i = 0; i < knnMatches21.size(); ++i) {
      if (knnMatches21[i].size() != 2)
        continue;

      if (knnMatches21[i][0].distance <
          ratioThresh * knnMatches21[i][1].distance) {
        result->map2_keypoints.push_back(
            slice2.kp[knnMatches21[i][0].queryIdx]);
        result->map1_keypoints.push_back(
            slice1.kp[knnMatches21[i][0].trainIdx]);
        result->distances.push_back(knnMatches21[i][0].distance);
      }
    }
  }

  return result;
}

MatchingResultPtr MapMatcherBase::MatchKeyPointsGMS(const Slice &slice1,
                                                    const Slice &slice2) const {
  MatchingResultPtr result(new MatchingResult());

  // Extract initial matches for GMS
  std::vector<cv::DMatch> putative_matches;

  // cv::Ptr<cv::DescriptorMatcher> matcher =
  //     cv::BFMatcher::create(cv::NORM_HAMMING, parameters_.cross_match);
  // matcher->match(slice1.desc, slice2.desc, putative_matches);

  slice2.matcher->match(slice1.desc, putative_matches);

  spdlog::debug("Num. putative matches: {}", putative_matches.size());

  // Refine matches with GMS
  std::vector<cv::DMatch> refined_matches;

  // With rotation, but without scale changes
  cv::xfeatures2d::matchGMS(slice1.binary_image.size(),
                            slice2.binary_image.size(), slice1.kp, slice2.kp,
                            putative_matches, refined_matches, true, false,
                            parameters_.gms_threshold_factor);
  spdlog::debug("Num. refined matches: {}", refined_matches.size());

  std::vector<cv::KeyPoint> kp1match, kp2match;

  for (const auto &match : refined_matches) {
    result->map1_keypoints.push_back(slice1.kp[match.queryIdx]);
    result->map2_keypoints.push_back(slice2.kp[match.trainIdx]);
    result->distances.push_back(match.distance);
  }

  return result;
}

Eigen::Matrix4d
MapMatcherBase::ConstructTransformFromParameters(double x, double y, double z,
                                                 double t) const {
  Eigen::Affine3d result_pose = Eigen::Affine3d::Identity();
  result_pose.translation() << x, y, z;
  result_pose.rotate(Eigen::AngleAxisd(t, Eigen::Vector3d::UnitZ()));
  Eigen::Matrix4d result_pose_mat = result_pose.matrix();
  return result_pose_mat;
}

void MapMatcherBase::VisualizeHypothesis(const PointCloud::Ptr &map1_pcd,
                                         const PointCloud::Ptr &map2_pcd,
                                         const HypothesisPtr &result) const {
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
    pcl::transformPointCloud(*(result->inlier_points_2),
                             *map2_inliers_transformed, result->pose);

    PointCloudColor map1_inliers_color(result->inlier_points_1, 255, 0, 255),
        map2_inliers_color(map2_inliers_transformed, 0, 255, 255);

    viewer.addPointCloud(result->inlier_points_1, map1_inliers_color,
                         "map1_inliers", vp2);
    viewer.addPointCloud(map2_inliers_transformed, map2_inliers_color,
                         "map2_inliers", vp2);
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

PointT MapMatcherBase::ComputeResultSpread(const HypothesisPtr &result) const {
  // Construct inliers pcds if they do not exist
  if (result->inlier_points_1 == nullptr ||
      result->inlier_points_2 == nullptr) {
    result->inlier_points_1 = PointCloud::Ptr(new PointCloud());
    result->inlier_points_2 = PointCloud::Ptr(new PointCloud());

    for (const SliceTransformPtr &tf_result : result->inlier_slices) {
      // Retrieve respective slice heights
      double height_map1 = tf_result->slice1->height,
             height_map2 = tf_result->slice2->height;

      for (const std::pair<cv::Point2f, cv::Point2f> &inlier_pair :
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

HypothesisPtr MapMatcherBase::RefineResult(const HypothesisPtr &result) const {
  HypothesisPtr result_refined(new Hypothesis(*result));
  Eigen::Matrix4d result_pose = result->pose;

  // ICP to be performed over feature points rather than all points
  PointCloud::Ptr inliers2_refinement(new PointCloud());
  pcl::transformPointCloud(*(result->inlier_points_2), *inliers2_refinement,
                           result_pose);
  Eigen::Matrix4f icp_result_f = Eigen::Matrix4f::Identity();

  if (parameters_.icp_refinement == 1) { // ICP
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputTarget(result->inlier_points_1);
    icp.setInputSource(inliers2_refinement);
    icp.setMaxCorrespondenceDistance(parameters_.grid_size * 2.0);
    icp.setUseReciprocalCorrespondences(true);
    PointCloud::Ptr resIcp(new PointCloud());
    icp.align(*resIcp);
    icp_result_f = icp.getFinalTransformation();

  } else if (parameters_.icp_refinement == 2) { // GICP
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> icp;
    icp.setInputTarget(result->inlier_points_1);
    icp.setInputSource(inliers2_refinement);
    icp.setMaxCorrespondenceDistance(parameters_.grid_size * 2.0);
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
