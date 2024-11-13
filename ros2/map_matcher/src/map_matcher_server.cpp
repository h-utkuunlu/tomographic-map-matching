#include <functional>
#include <memory>
#include <thread>

#include "conversions.hpp"
#include <map_matcher_interfaces/action/match_point_cloud_maps.hpp>
#include <map_matcher_interfaces/action/match_slice_maps.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace map_matcher_ros {

class MatcherServer : public rclcpp::Node
{
public:
  using MatchSliceMaps = map_matcher_interfaces::action::MatchSliceMaps;
  using GoalHandleMatchSliceMaps = rclcpp_action::ServerGoalHandle<MatchSliceMaps>;

  explicit MatcherServer(const rclcpp::NodeOptions& options)
    : rclcpp::Node("map_matcher_server", options)
  {
    using namespace std::placeholders;

    this->declare_parameter("map_path", "");
    this->declare_parameter("map_topic", "cloud_map");
    this->declare_parameter("grid_topic", "map");

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
                map_topic = this->get_parameter("map_topic").as_string(),
                grid_topic = this->get_parameter("grid_topic").as_string();

    if (map_path.empty()) {
      map_subscription_ptr_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        map_topic, 10, std::bind(&MatcherServer::MapCallback, this, _1));
      grid_subscription_ptr_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        grid_topic, 10, std::bind(&MatcherServer::GridCallback, this, _1));
      RCLCPP_INFO(this->get_logger(),
                  "Listening to the map messages on '%s' (pcd) and '%s' (grid)",
                  map_topic.c_str(),
                  grid_topic.c_str());
    } else {
      map_matcher::PointCloud::Ptr map_pcd(new map_matcher::PointCloud());
      pcl::io::loadPCDFile(map_path, *map_pcd);

      local_map_ = consensus_matcher_->ComputeSliceImages(map_pcd);
      consensus_matcher_->ComputeSliceFeatures(local_map_);

      RCLCPP_INFO(this->get_logger(), "Loaded map located at '%s'", map_path.c_str());
    }

    // Matching using sliced maps
    {
      auto handle_goal = [this](const rclcpp_action::GoalUUID& uuid,
                                std::shared_ptr<const MatchSliceMaps::Goal> goal) {
        RCLCPP_INFO(this->get_logger(), "Received goal request");
        (void)uuid;
        (void)goal;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      };

      auto handle_cancel =
        [this](const std::shared_ptr<GoalHandleMatchSliceMaps> goal_handle) {
          RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
          (void)goal_handle;
          return rclcpp_action::CancelResponse::ACCEPT;
        };

      auto handle_accepted =
        [this](const std::shared_ptr<GoalHandleMatchSliceMaps> goal_handle) {
          auto execute_in_thread = [this, goal_handle]() {
            return this->execute(goal_handle);
          };
          std::thread{ execute_in_thread }.detach();
        };

      this->server_ptr_ = rclcpp_action::create_server<MatchSliceMaps>(
        this, "map_matching_action", handle_goal, handle_cancel, handle_accepted);
    }
  }

private:
  rclcpp_action::Server<MatchSliceMaps>::SharedPtr server_ptr_;

  void MapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {

    RCLCPP_INFO(this->get_logger(), "Received new map");

    map_matcher::PointCloud::Ptr new_map_pcd(new map_matcher::PointCloud());
    pcl::fromROSMsg(*msg, *new_map_pcd);
    auto new_map = consensus_matcher_->ComputeSliceImages(new_map_pcd);
    consensus_matcher_->ComputeSliceFeatures(new_map);
    local_map_ = new_map;
    RCLCPP_INFO(this->get_logger(), "Map processing complete");
    return;
  }

  void GridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {

    RCLCPP_INFO(this->get_logger(), "Received new grid map");
    local_grid_ = *msg;
    return;
  }

  void execute(const std::shared_ptr<GoalHandleMatchSliceMaps> goal_handle)
  {

    auto feedback = std::make_shared<MatchSliceMaps::Feedback>();
    auto result = std::make_shared<MatchSliceMaps::Result>();
    const auto goal = goal_handle->get_goal();

    // TODO: Add enum codes for status here
    RCLCPP_DEBUG(this->get_logger(),
                 "Received map. Num. slices in map: %lu",
                 goal->map.sliced_map.size());

    feedback->status = "Conversion";
    goal_handle->publish_feedback(feedback);

    // ROS -> vector<SlicePtr>
    std::vector<map_matcher::SlicePtr> source_map;
    ConvertFromROS(goal->map, source_map);

    RCLCPP_DEBUG(
      this->get_logger(), "Converted source map. Num. slices: %lu", source_map.size());

    feedback->status = "Registration";
    goal_handle->publish_feedback(feedback);
    if (goal_handle->is_canceling()) {
      goal_handle->canceled(result);
      RCLCPP_WARN(this->get_logger(), "Matching canceled");
      return;
    }

    std::vector<map_matcher::HypothesisPtr> results_all =
      consensus_matcher_->CorrelateSlices(source_map, local_map_);

    feedback->status = "Parsing";
    goal_handle->publish_feedback(feedback);
    if (goal_handle->is_canceling()) {
      goal_handle->canceled(result);
      RCLCPP_WARN(this->get_logger(), "Matching canceled");
      return;
    }

    Eigen::Affine3d pose_affine;
    pose_affine.matrix() = results_all[0]->pose;

    if (rclcpp::ok()) {
      result->map_pose = tf2::toMsg(pose_affine);
      result->grid_map = local_grid_;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Matching complete");
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_subscription_ptr_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_subscription_ptr_;
  std::unique_ptr<map_matcher::Consensus> consensus_matcher_;
  std::vector<map_matcher::SlicePtr> local_map_;
  nav_msgs::msg::OccupancyGrid local_grid_;
};

} // namespace map_matcher_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_matcher_ros::MatcherServer);
