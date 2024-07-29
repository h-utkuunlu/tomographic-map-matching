#pragma once

#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

typedef pcl::PointXYZI KeypointT;
typedef pcl::Normal NormalT;
typedef pcl::FPFHSignature33 FeatureT;

typedef pcl::PointCloud<KeypointT> KeypointCloud;
typedef pcl::PointCloud<NormalT> NormalCloud;
typedef pcl::PointCloud<FeatureT> FeatureCloud;

typedef pcl::HarrisKeypoint3D<PointT, KeypointT> KeypointDetector;

class FPFHRANSAC : public MapMatcherBase
{

public:
  FPFHRANSAC();
  FPFHRANSAC(const json& parameters);
  void GetParameters(json& output) const override;
  void UpdateParameters(const json& input) override;
  std::string GetName() const override { return "FPFH-RANSAC"; }

  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                                       const PointCloud::Ptr pcd2,
                                       json& stats) const override;
  void VisualizeKeypoints(const PointCloud::Ptr pcd,
                          const PointCloud::Ptr keypoints) const;

private:
  void ExtractInlierKeypoints(const PointCloud::Ptr map1_pcd,
                              const PointCloud::Ptr map2_pcd,
                              const pcl::CorrespondencesPtr correspondences,
                              PointCloud::Ptr map1_inliers,
                              PointCloud::Ptr map2_inliers) const;
  void DetectAndDescribeKeypoints(const PointCloud::Ptr input,
                                  PointCloud::Ptr keypoints,
                                  FeatureCloud::Ptr features) const;

  float normal_radius_ = 0.3;
  float keypoint_radius_ = 0.2;
  int response_method_ = 1;
  float corner_threshold_ = 0.0;
  float descriptor_radius_ = 0.5;
  float ransac_inlier_threshold_ = 0.1;
  bool ransac_refine_model_ = true;
};

} // namespace map_matcher
