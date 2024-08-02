#pragma once

#include <teaser/registration.h>
#include <tomographic_map_matching/tomographic_matcher.hpp>

namespace map_matcher {

class ORBTEASER : public TomographicMatcher
{

public:
  ORBTEASER();
  ORBTEASER(const json& parameters);
  std::string GetName() const override { return "ORB-TEASER"; }
  void UpdateParameters(const json& parameters) override;
  void GetParameters(json& output) const override;
  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr source,
                                       const PointCloud::Ptr target,
                                       json& stats) const override;

private:
  std::vector<HypothesisPtr> CorrelateSlices(
    const std::vector<SlicePtr>& source_features,
    const std::vector<SlicePtr>& target_features) const;
  HypothesisPtr RunTeaserWith3DMatches(
    const std::vector<SlicePtr>& source_features,
    const std::vector<SlicePtr>& target_features) const;
  HypothesisPtr RegisterForGivenInterval(const std::vector<SlicePtr>& source,
                                         const std::vector<SlicePtr>& target,
                                         HeightIndices indices) const;
  std::shared_ptr<teaser::RobustRegistrationSolver> RegisterPointsWithTeaser(
    const PointCloud::Ptr pcd1,
    const PointCloud::Ptr pcd2) const;
  void SelectTopNMatches(PointCloud::Ptr& source_points,
                         PointCloud::Ptr& target_points,
                         const std::vector<float>& distances) const;
  HypothesisPtr ConstructSolutionFromSolverState(
    const std::shared_ptr<teaser::RobustRegistrationSolver>& solver,
    const PointCloud::Ptr& source_points,
    const PointCloud::Ptr& target_points) const;

  double teaser_noise_bound_ = 0.02;
  size_t teaser_num_correspondences_max_ = 10000;
  bool teaser_verbose_ = false;
  bool teaser_3d_ = false;
};
} // namespace map_matcher
