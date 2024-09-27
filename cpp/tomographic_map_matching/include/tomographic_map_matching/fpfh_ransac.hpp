#pragma once

#include <tomographic_map_matching/fpfh_base.hpp>

namespace map_matcher {

class FPFHRANSAC : public FPFHBase
{

public:
  FPFHRANSAC();
  FPFHRANSAC(const json& parameters);
  void GetParameters(json& output) const override;
  void UpdateParameters(const json& input) override;
  std::string GetName() const override { return "FPFH-RANSAC"; }

  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr source,
                                       const PointCloud::Ptr target,
                                       json& stats) const override;

private:
  float ransac_inlier_threshold_ = 0.1;
  bool ransac_refine_model_ = true;
};

} // namespace map_matcher
