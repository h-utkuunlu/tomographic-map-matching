#include <functional>
#include <memory>
#include <thread>

#include "tomographic_map_matching/consensus.hpp"
#include <map_matcher_interfaces/action/match_maps.hpp>
#include <map_matcher_interfaces/srv/trigger_matching.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace map_matcher_ros {

class MatcherClient : public rclcpp::Node
{
public:
  using MatchMaps = map_matcher_interfaces::action::MatchMaps;
  using GoalHandleMatchMaps = rclcpp_action::ClientGoalHandle<MatchMaps>;
  using TriggerMatching = map_matcher_interfaces::srv::TriggerMatching;

  explicit MatcherClient(const rclcpp::NodeOptions& options)
    : rclcpp::Node("map_matcher_client", options)
  {

    using namespace std::placeholders;

    // Service to trigger initializing an action server
    trigger_srv_ = this->create_service<TriggerMatching>(
      "trigger_matching",
      std::bind(&MatcherClient::TriggerServiceHandle, this, _1, _2));
  }

  void SendGoal() { return; }

  void TriggerServiceHandle(const std::shared_ptr<TriggerMatching::Request> request,
                            std::shared_ptr<TriggerMatching::Response> response)
  {
    RCLCPP_INFO(this->get_logger(),
                "Initiating a matcher client targeting robot ID %s",
                request->target_id.c_str());

    // Initialize action client
    //   this->client_ptr_ = rclcpp_action::create_client<MatchMaps>(this,
    //   "matcher_action");

    RCLCPP_INFO(this->get_logger(),
                "Initiated matcher client targeting %s successfully",
                request->target_id.c_str());
  }

private:
  rclcpp_action::Client<MatchMaps>::SharedPtr client_ptr_;
  rclcpp::Service<TriggerMatching>::SharedPtr trigger_srv_;
};
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_matcher_ros::MatcherClient);
