#pragma once

#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

struct ConsensusParameters : public Parameters {

  ConsensusParameters() = default;
  ConsensusParameters(const ConsensusParameters &) = default;
  ConsensusParameters(const Parameters &p) : Parameters(p) {}

  double consensus_ransac_factor = 5.0;
  bool consensus_use_rigid = false;
};

void to_json(json &j, const ConsensusParameters &p);
void from_json(const json &j, ConsensusParameters &p);

class Consensus : public MapMatcherBase {
public:
  Consensus();
  Consensus(ConsensusParameters parameters);
  json GetParameters() const override;
  void SetParameters(const json &parameters);
  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                                       const PointCloud::Ptr pcd2,
                                       json &stats) const override;
  std::string GetName() const override { return "Consensus"; }

private:
  ConsensusParameters parameters_;
  std::vector<HypothesisPtr>
  CorrelateSlices(const std::vector<SlicePtr> &map1_features,
                  const std::vector<SlicePtr> &map2_features) const;
  std::vector<SliceTransformPtr> ComputeMapTf(const std::vector<SlicePtr> &map1,
                                              const std::vector<SlicePtr> &map2,
                                              HeightIndices indices) const;
  HypothesisPtr
  VoteBetweenSlices(const std::vector<SliceTransformPtr> &results) const;
};

} // namespace map_matcher
