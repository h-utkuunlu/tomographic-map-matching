#include <functional>
#include <memory>
#include <thread>

#include "conversions.hpp"
#include <map_matcher_interfaces/action/match_maps.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

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

    this->declare_parameter("map_path", "");
    this->declare_parameter("map_topic", "cloud_map");

    // Set up matcher
    map_matcher::json matcher_parameters;
    matcher_parameters["algorithm"] = 0;
    matcher_parameters["grid_size"] = 0.05;
    matcher_parameters["slice_z_height"] = 0.05;
    matcher_parameters["cross_match"] = true;
    matcher_parameters["consensus_use_rigid"] = true;
    consensus_matcher_ = std::make_unique<map_matcher::Consensus>(matcher_parameters);

    // Set up map subscription / load map
    std::string map_path = this->get_parameter("map_path").as_string(),
                map_topic = this->get_parameter("map_topic").as_string();

    if (map_path.empty()) {
      map_subscription_ptr_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        map_topic, 10, std::bind(&MatcherServer::MapCallback, this, _1));
      RCLCPP_INFO(
        this->get_logger(), "Listening to the map messages on '%s'", map_topic.c_str());
    } else {
      map_matcher::PointCloud::Ptr map_pcd(new map_matcher::PointCloud());
      pcl::io::loadPCDFile(map_path, *map_pcd);

      local_map_ = consensus_matcher_->ComputeSliceImages(map_pcd);
      consensus_matcher_->ComputeSliceFeatures(local_map_);

      RCLCPP_INFO(this->get_logger(), "Loaded map located at '%s'", map_path.c_str());
    }

    this->server_ptr_ = rclcpp_action::create_server<MatchMaps>(
      this,
      "map_matching_action",
      std::bind(&MatcherServer::handle_goal, this, _1, _2),
      std::bind(&MatcherServer::handle_cancel, this, _1),
      std::bind(&MatcherServer::handle_accepted, this, _1));
  };

private:
  rclcpp_action::Server<MatchMaps>::SharedPtr server_ptr_;

  void MapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {

    RCLCPP_INFO(this->get_logger(), "Received new map");

    map_matcher::PointCloud::Ptr new_map_pcd(new map_matcher::PointCloud());
    pcl::fromROSMsg(*msg, *new_map_pcd);
    auto new_map = consensus_matcher_->ComputeSliceImages(new_map_pcd);
    consensus_matcher_->ComputeSliceFeatures(new_map);
    local_map_ = new_map;

    return;
  }

  rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
                                          std::shared_ptr<const MatchMaps::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request");
    (void)uuid;
    (void)goal;
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

    std::vector<map_matcher::SlicePtr> source_map;
    const auto goal = goal_handle->get_goal();

    ConvertFromROS(goal->map, source_map);

    std::vector<map_matcher::HypothesisPtr> results_all =
      consensus_matcher_->CorrelateSlices(source_map, local_map_);

    // TODO: Parser results_all[0] to result

    goal_handle->succeed(result);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_subscription_ptr_;
  std::unique_ptr<map_matcher::Consensus> consensus_matcher_;
  std::vector<map_matcher::SlicePtr> local_map_;
};
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_matcher_ros::MatcherServer);
