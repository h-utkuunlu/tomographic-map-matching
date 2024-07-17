#pragma once

#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <teaser/registration.h>
#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

typedef pcl::PointXYZI KeypointT;
typedef pcl::Normal NormalT;
typedef pcl::FPFHSignature33 FeatureT;

typedef pcl::PointCloud<KeypointT> KeypointCloud;
typedef pcl::PointCloud<NormalT> NormalCloud;
typedef pcl::PointCloud<FeatureT> FeatureCloud;

typedef pcl::HarrisKeypoint3D<PointT, KeypointT> KeypointDetector;

struct FPFHTEASERParameters : public Parameters {
  FPFHTEASERParameters() = default;
  FPFHTEASERParameters(const FPFHTEASERParameters &) = default;
  FPFHTEASERParameters(const Parameters &p) : Parameters(p) {}
  float normal_radius = 0.3;
  float keypoint_radius = 0.2;
  int response_method = 1;
  float corner_threshold = 0.0;
  float descriptor_radius = 0.5;
  double teaser_noise_bound = 0.02;
  size_t teaser_num_correspondences_max = 10000;
  bool teaser_verbose = false;
};

void to_json(json &j, const FPFHTEASERParameters &p);
void from_json(const json &j, FPFHTEASERParameters &p);

class FPFHTEASER : public MapMatcherBase {
public:
  FPFHTEASER();
  FPFHTEASER(FPFHTEASERParameters parameters);
  json GetParameters() const override;
  void SetParameters(const json &parameters);
  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                                       const PointCloud::Ptr pcd2,
                                       json &stats) const override;
  void VisualizeKeypoints(const PointCloud::Ptr pcd,
                          const PointCloud::Ptr keypoints) const;
  std::string GetName() const override { return "FPFH-TEASER"; }

private:
  FPFHTEASERParameters parameters_;
  void ExtractInlierKeypoints(const PointCloud::Ptr map1_pcd,
                              const PointCloud::Ptr map2_pcd,
                              const pcl::CorrespondencesPtr correspondences,
                              PointCloud::Ptr map1_inliers,
                              PointCloud::Ptr map2_inliers) const;
  void DetectAndDescribeKeypoints(const PointCloud::Ptr input,
                                  PointCloud::Ptr keypoints,
                                  FeatureCloud::Ptr features) const;
};

} // namespace map_matcher
