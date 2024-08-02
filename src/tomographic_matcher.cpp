#include <opencv2/xfeatures2d.hpp>
#include <tomographic_map_matching/tomographic_matcher.hpp>

namespace map_matcher {

TomographicMatcher::TomographicMatcher()
  : MapMatcherBase()
{
}

TomographicMatcher::TomographicMatcher(const json parameters)
  : MapMatcherBase(parameters)
{
  UpdateParameters(parameters);
}

void
TomographicMatcher::UpdateParameters(const json& input)
{
  UpdateSingleParameter(input, "cross_match", cross_match_);
  UpdateSingleParameter(input, "slice_z_height", slice_z_height_);
  UpdateSingleParameter(input, "orb_num_features", orb_num_features_);
  UpdateSingleParameter(input, "orb_n_levels", orb_n_levels_);
  UpdateSingleParameter(input, "orb_edge_threshold", orb_edge_threshold_);
  UpdateSingleParameter(input, "orb_first_level", orb_first_level_);
  UpdateSingleParameter(input, "orb_wta_k", orb_wta_k_);
  UpdateSingleParameter(input, "orb_patch_size", orb_patch_size_);
  UpdateSingleParameter(input, "orb_fast_threshold", orb_fast_threshold_);
  UpdateSingleParameter(input, "gms_matching", gms_matching_);
  UpdateSingleParameter(input, "gms_threshold_factor", gms_threshold_factor_);
}

void
TomographicMatcher::GetParameters(json& output) const
{
  MapMatcherBase::GetParameters(output);
  output["cross_match"] = cross_match_;
  output["slice_z_height"] = slice_z_height_;
  output["orb_num_features"] = orb_num_features_;
  output["orb_n_levels"] = orb_n_levels_;
  output["orb_edge_threshold"] = orb_edge_threshold_;
  output["orb_first_level"] = orb_first_level_;
  output["orb_wta_k"] = orb_wta_k_;
  output["orb_patch_size"] = orb_patch_size_;
  output["orb_fast_threshold"] = orb_fast_threshold_;
  output["gms_matching"] = gms_matching_;
  output["gms_threshold_factor"] = gms_threshold_factor_;
}

PointCloud::Ptr
TomographicMatcher::ExtractSlice(const PointCloud::Ptr& pcd, double height) const
{
  size_t N = pcd->size();
  double hmin = height - (slice_z_height_ / 2.0),
         hmax = height + (slice_z_height_ / 2.0);
  PointCloud::Ptr slice(new PointCloud());

  for (size_t i = 0; i < N; ++i) {
    const PointT& pt = (*pcd)[i];
    if (pt.z > hmin && pt.z < hmax) {
      slice->push_back(PointT(pt.x, pt.y, 0.0));
    }
  }

  return slice;
}

void
TomographicMatcher::ConvertPCDSliceToImage(const PointCloud::Ptr& pcd_slice,
                                           Slice& image_slice) const
{
  size_t yPix = static_cast<size_t>(image_slice.binary_image.rows);
  // size_t xPix = static_cast<size_t>(image_slice.binary_image.cols);

  for (const auto& pt : *pcd_slice) {
    // Find coordinate of the point on the image
    size_t xIdx = std::round((pt.x - image_slice.slice_bounds.lower.x) / grid_size_);
    size_t yIdx = std::round((pt.y - image_slice.slice_bounds.lower.y) / grid_size_);

    // Flip y direction to match conventional image representation (y up, x
    // right)
    yIdx = yPix - yIdx - 1;
    image_slice.binary_image.at<uchar>(yIdx, xIdx) = 255;
  }

  // Median filter if selected
  if (median_filter_) {
    cv::medianBlur(image_slice.binary_image, image_slice.binary_image, 3);
  }
}

std::vector<cv::Point2f>
TomographicMatcher::img2real(const std::vector<cv::Point2f>& pts,
                             const CartesianBounds& mapBounds) const
{
  // Needed to flip direction of y
  size_t yPix = std::round((mapBounds.upper.y - mapBounds.lower.y) / grid_size_ + 1.0);

  std::vector<cv::Point2f> converted(pts.size());
  size_t index = 0;

  for (const auto& pt : pts) {
    float x = pt.x * grid_size_ + mapBounds.lower.x;
    float y = (yPix - pt.y - 1) * grid_size_ + mapBounds.lower.y;
    converted[index++] = cv::Point2f(x, y);
  }

  return converted;
}

PointCloud::Ptr
TomographicMatcher::img2real(const std::vector<cv::Point2f>& pts,
                             const CartesianBounds& mapBounds,
                             double z_height) const
{
  // Needed to flip direction of y
  size_t yPix = std::round((mapBounds.upper.y - mapBounds.lower.y) / grid_size_ + 1.0);

  PointCloud::Ptr converted(new PointCloud());

  for (const auto& pt : pts) {
    float x = pt.x * grid_size_ + mapBounds.lower.x;
    float y = (yPix - pt.y - 1) * grid_size_ + mapBounds.lower.y;
    converted->push_back(PointT(x, y, z_height));
  }

  return converted;
}

std::vector<SlicePtr>
TomographicMatcher::ComputeSliceImages(const PointCloud::Ptr& map) const
{
  CartesianBounds map_bounds = CalculateBounds(map);

  // Number of pixels in x- and y-directions are independent of individual point
  // cloud slices. Calculate it once
  size_t xPix =
    std::round((map_bounds.upper.x - map_bounds.lower.x) / grid_size_ + 1.0);
  size_t yPix =
    std::round((map_bounds.upper.y - map_bounds.lower.y) / grid_size_ + 1.0);

  spdlog::debug("Slice dimensions: ({}, {})", xPix, yPix);

  // Identify the number of slices that can be computed from this particular map
  double zmin = map_bounds.lower.z, zmax = map_bounds.upper.z;
  size_t maximum_index =
    static_cast<size_t>(std::round((zmax - zmin) / grid_size_) + 1);

  std::vector<SlicePtr> image_slices(maximum_index);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < maximum_index; ++i) {
    // Unified slice bounds for images to be the same size across different
    // layers
    double height = zmin + i * grid_size_;
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
  image_slices_trimmed.insert(image_slices_trimmed.end(),
                              std::make_move_iterator(image_slices.begin() + start),
                              std::make_move_iterator(image_slices.end() - end));

  return image_slices_trimmed;
}

void
TomographicMatcher::VisualizeHypothesisSlices(const HypothesisPtr hypothesis) const
{
  size_t num_inlier_slices = hypothesis->inlier_slices.size(), current_idx = 0;

  if (num_inlier_slices == 0) {
    spdlog::warn("There are no inlier slices for the hypothesis. Cannot visualize");
    return;
  }

  cv::namedWindow("target_slice", cv::WINDOW_NORMAL);
  cv::namedWindow("source_slice", cv::WINDOW_NORMAL);
  // cv::namedWindow("combined", cv::WINDOW_NORMAL);

  // Interactive visualization
  while (true) {
    spdlog::info("Pair {} / {}", current_idx + 1, num_inlier_slices);

    SliceTransformPtr& slice_pair = hypothesis->inlier_slices[current_idx];
    VisualizeSlice(slice_pair->target_slice, "target_slice");
    VisualizeSlice(slice_pair->source_slice, "source_slice");

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

void
TomographicMatcher::VisualizeSlice(const SlicePtr slice, std::string window_name) const
{
  cv::Mat display_image;

  if (slice->is_descriptive) {
    // Single channel to 3 channels for color-coding
    cv::cvtColor(slice->binary_image, display_image, cv::COLOR_GRAY2BGR);
    cv::imshow(window_name, display_image);
  }
}

std::vector<SlicePtr>&
TomographicMatcher::ComputeSliceFeatures(std::vector<SlicePtr>& image_slices) const
{
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < image_slices.size(); ++i) {
    cv::Ptr<cv::ORB> detector = cv::ORB::create(orb_num_features_,
                                                orb_scale_factor_,
                                                orb_n_levels_,
                                                orb_edge_threshold_,
                                                orb_first_level_,
                                                orb_wta_k_,
                                                cv::ORB::HARRIS_SCORE,
                                                orb_patch_size_,
                                                orb_fast_threshold_);
    Slice& image_slice = *image_slices[i];
    size_t padding = detector->getEdgeThreshold();
    double padding_d = static_cast<double>(padding);

    // Do not waste time on slices that do not have enough points
    if (image_slice.is_descriptive) {
      cv::Mat padded_image;
      cv::copyMakeBorder(image_slice.binary_image,
                         padded_image,
                         padding,
                         padding,
                         padding,
                         padding,
                         cv::BORDER_CONSTANT);

      detector->detectAndCompute(
        padded_image, cv::noArray(), image_slice.kp, image_slice.desc);

      // Subtract the padded coordinates
      for (auto& kp : image_slice.kp) {
        kp.pt.x -= padding_d;
        kp.pt.y -= padding_d;
      }

      // Matcher is embedded into the slice for faster querying
      if (approximate_neighbors_) {
        image_slice.matcher = cv::makePtr<cv::FlannBasedMatcher>(
          cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(
            lsh_num_tables_, lsh_key_size_, lsh_multiprobe_level_)));
      } else {
        if (orb_wta_k_ == 2) {
          image_slice.matcher = cv::BFMatcher::create(cv::NORM_HAMMING, cross_match_);
        } else if (orb_wta_k_ == 3 or orb_wta_k_ == 4) {
          image_slice.matcher = cv::BFMatcher::create(cv::NORM_HAMMING2, cross_match_);
        } else {
          spdlog::critical("ORB WTA_K cannot be anything other than 2, 3, or "
                           "4. Set value: {}",
                           orb_wta_k_);
        }
      }
      image_slice.matcher->add(image_slice.desc);
      image_slice.matcher->train();
    }
  }
  return image_slices;
}

MatchingResultPtr
TomographicMatcher::MatchKeyPoints(const Slice& source_slice,
                                   const Slice& target_slice) const
{
  MatchingResultPtr result(new MatchingResult());

  // Skip if not descriptive
  if (!(target_slice.is_descriptive && source_slice.is_descriptive))
    return result;

  if (cross_match_) {
    // Cross-matching
    std::vector<cv::DMatch> matches;
    source_slice.matcher->match(target_slice.desc, matches);

    for (const auto& match : matches) {
      result->target_keypoints.push_back(target_slice.kp[match.queryIdx]);
      result->source_keypoints.push_back(source_slice.kp[match.trainIdx]);
      result->distances.push_back(match.distance);
    }

  } else {
    // 2-way match with ratio test
    std::vector<std::vector<cv::DMatch>> knnMatches12, knnMatches21;
    source_slice.matcher->knnMatch(target_slice.desc, knnMatches12, 2);
    target_slice.matcher->knnMatch(source_slice.desc, knnMatches21, 2);

    const float ratioThresh = 0.7f;

    for (size_t i = 0; i < knnMatches12.size(); ++i) {
      if (knnMatches12[i].size() != 2)
        continue;

      if (knnMatches12[i][0].distance < ratioThresh * knnMatches12[i][1].distance) {
        result->target_keypoints.push_back(
          target_slice.kp[knnMatches12[i][0].queryIdx]);
        result->source_keypoints.push_back(
          source_slice.kp[knnMatches12[i][0].trainIdx]);
        result->distances.push_back(knnMatches12[i][0].distance);
      }
    }

    for (size_t i = 0; i < knnMatches21.size(); ++i) {
      if (knnMatches21[i].size() != 2)
        continue;

      if (knnMatches21[i][0].distance < ratioThresh * knnMatches21[i][1].distance) {
        result->source_keypoints.push_back(
          source_slice.kp[knnMatches21[i][0].queryIdx]);
        result->target_keypoints.push_back(
          target_slice.kp[knnMatches21[i][0].trainIdx]);
        result->distances.push_back(knnMatches21[i][0].distance);
      }
    }
  }

  return result;
}

MatchingResultPtr
TomographicMatcher::MatchKeyPointsGMS(const Slice& source_slice,
                                      const Slice& target_slice) const
{
  MatchingResultPtr result(new MatchingResult());

  // Extract initial matches for GMS
  std::vector<cv::DMatch> putative_matches;

  // cv::Ptr<cv::DescriptorMatcher> matcher =
  //     cv::BFMatcher::create(cv::NORM_HAMMING, parameters_.cross_match);
  // matcher->match(target_slice.desc, source_slice.desc, putative_matches);

  source_slice.matcher->match(target_slice.desc, putative_matches);

  spdlog::debug("Num. putative matches: {}", putative_matches.size());

  // Refine matches with GMS
  std::vector<cv::DMatch> refined_matches;

  // With rotation, but without scale changes
  cv::xfeatures2d::matchGMS(target_slice.binary_image.size(),
                            source_slice.binary_image.size(),
                            target_slice.kp,
                            source_slice.kp,
                            putative_matches,
                            refined_matches,
                            true,
                            false,
                            gms_threshold_factor_);
  spdlog::debug("Num. refined matches: {}", refined_matches.size());

  std::vector<cv::KeyPoint> kp1match, kp2match;

  for (const auto& match : refined_matches) {
    result->target_keypoints.push_back(target_slice.kp[match.queryIdx]);
    result->source_keypoints.push_back(source_slice.kp[match.trainIdx]);
    result->distances.push_back(match.distance);
  }

  return result;
}

} // namespace map_matcher
