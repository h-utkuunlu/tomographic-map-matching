#include "map_matcher_node.hpp"

namespace tomographic_map_matcher_node {

MatcherNode::MatcherNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("map_matcher_node", options)
{
  matcher_ = std::make_unique<map_matcher::Consensus>();
  RCLCPP_INFO(this->get_logger(), "Map matcher ROS 2 node initialized");
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tomographic_map_matcher_node::MatcherNode)
