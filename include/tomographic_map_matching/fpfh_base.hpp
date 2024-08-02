#pragma once

#include <pcl/features/normal_3d_omp.h>
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

class FPFHBase : public MapMatcherBase
{

public:
  virtual void UpdateParameters(const json& input) override;
  virtual void GetParameters(json& output) const override;
  void VisualizeKeypoints(const PointCloud::Ptr pcd,
                          const PointCloud::Ptr keypoints) const;

protected:
  FPFHBase();
  FPFHBase(const json& parameters);
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
};

} // namespace map_matcher
