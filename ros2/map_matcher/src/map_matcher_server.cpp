#include <functional>
#include <memory>
#include <thread>

#include "tomographic_map_matching/consensus.hpp"
#include <map_matcher_interfaces/action/match_maps.hpp>
#include <map_matcher_interfaces/msg/slice_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace map_matcher_ros {

class MatcherServer : public rclcpp::Node
{
public:
  using MatchMaps = map_matcher_interfaces::action::MatchMaps;
  using GoalHandleMatchMaps = rclcpp_action::ServerGoalHandle<MatchMaps>;

  explicit MatcherServer(const rclcpp::NodeOptions& options)
    : rclcpp::Node("map_matcher_server", options)
  {
    using namespace std::placeholders;

    this->server_ptr_ = rclcpp_action::create_server<MatchMaps>(
      this,
      "matcher_action",
      std::bind(&MatcherServer::handle_goal, this, _1, _2),
      std::bind(&MatcherServer::handle_cancel, this, _1),
      std::bind(&MatcherServer::handle_accepted, this, _1));
  };

private:
  rclcpp_action::Server<MatchMaps>::SharedPtr server_ptr_;

  rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
                                          std::shared_ptr<const MatchMaps::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request");
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleMatchMaps> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleMatchMaps> goal_handle)
  {
    using namespace std::placeholders;
    std::thread{ std::bind(&MatcherServer::execute, this, _1), goal_handle }.detach();
  }

  void execute(const std::shared_ptr<GoalHandleMatchMaps> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal");

    auto feedback = std::make_shared<MatchMaps::Feedback>();
    auto result = std::make_shared<MatchMaps::Result>();

    // TODO: Add enum codes for status here
    feedback->status = 0;
    goal_handle->publish_feedback(feedback);

    // Execution code here

    goal_handle->succeed(result);
  }
};
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_matcher_ros::MatcherServer);
