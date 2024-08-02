#pragma once

#include <teaser/registration.h>
#include <tomographic_map_matching/fpfh_base.hpp>

namespace map_matcher {

class FPFHTEASER : public FPFHBase
{
public:
  FPFHTEASER();
  FPFHTEASER(const json& parameters);
  void GetParameters(json& output) const override;
  void UpdateParameters(const json& input) override;
  std::string GetName() const override { return "FPFH-TEASER"; }

  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr source,
                                       const PointCloud::Ptr target,
                                       json& stats) const override;

private:
  double teaser_noise_bound_ = 0.02;
  size_t teaser_num_correspondences_max_ = 10000;
  bool teaser_verbose_ = false;
};

} // namespace map_matcher
