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
  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                                       const PointCloud::Ptr pcd2,
                                       json& stats) const override;
  std::string GetName() const override { return "Consensus"; }

private:
  std::vector<HypothesisPtr> CorrelateSlices(
    const std::vector<SlicePtr>& map1_features,
    const std::vector<SlicePtr>& map2_features) const;
  std::vector<SliceTransformPtr> ComputeMapTf(const std::vector<SlicePtr>& map1,
                                              const std::vector<SlicePtr>& map2,
                                              HeightIndices indices) const;
  HypothesisPtr VoteBetweenSlices(
    const std::vector<SliceTransformPtr>& results) const;

  double consensus_ransac_factor_ = 5.0;
  bool consensus_use_rigid_ = false;
};

} // namespace map_matcher
