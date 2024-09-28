#pragma once

#include "tomographic_map_matching/consensus.hpp"
#include <rclcpp/rclcpp.hpp>

namespace map_matcher_node {

class MatcherNode : public rclcpp::Node
{
public:
  MatcherNode() = delete;
  explicit MatcherNode(const rclcpp::NodeOptions& options);

private:
  std::unique_ptr<map_matcher::Consensus> matcher_;
};
}
