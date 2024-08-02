#pragma once

#include <tomographic_map_matching/tomographic_matcher.hpp>

namespace map_matcher {

class Consensus : public TomographicMatcher
{
public:
  Consensus();
  Consensus(const json& parameters);
  void UpdateParameters(const json& input) override;
  void GetParameters(json& output) const override;
  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr source,
                                       const PointCloud::Ptr target,
                                       json& stats) const override;
  std::string GetName() const override { return "Consensus"; }

private:
  std::vector<HypothesisPtr> CorrelateSlices(
    const std::vector<SlicePtr>& source_features,
    const std::vector<SlicePtr>& target_features) const;
  std::vector<SliceTransformPtr> ComputeMapTf(const std::vector<SlicePtr>& source,
                                              const std::vector<SlicePtr>& target,
                                              HeightIndices indices) const;
  HypothesisPtr VoteBetweenSlices(const std::vector<SliceTransformPtr>& results) const;

  double consensus_ransac_factor_ = 5.0;
  bool consensus_use_rigid_ = false;
};

} // namespace map_matcher
