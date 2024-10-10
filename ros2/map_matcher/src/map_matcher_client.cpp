#include <functional>
#include <memory>
#include <thread>

#include "conversions.hpp"
#include <map_matcher_interfaces/action/match_maps.hpp>
#include <map_matcher_interfaces/srv/trigger_matching.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

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

    // Parameters
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
        map_topic, 10, std::bind(&MatcherClient::MapCallback, this, _1));
      RCLCPP_INFO(
        this->get_logger(), "Listening to the map messages on '%s'", map_topic.c_str());
    } else {
      map_matcher::PointCloud::Ptr map_pcd(new map_matcher::PointCloud());
      pcl::io::loadPCDFile(map_path, *map_pcd);

      local_map_ = consensus_matcher_->ComputeSliceImages(map_pcd);
      consensus_matcher_->ComputeSliceFeatures(local_map_);

      RCLCPP_INFO(this->get_logger(), "Loaded map located at '%s'", map_path.c_str());
    }

    // Service to trigger initializing an action server
    srv_ptr_ = this->create_service<TriggerMatching>(
      "trigger_matching",
      std::bind(&MatcherClient::TriggerServiceHandle, this, _1, _2));
  }

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

  void TriggerServiceHandle(const std::shared_ptr<TriggerMatching::Request> request,
                            std::shared_ptr<TriggerMatching::Response> response)
  {
    const std::string action_destination =
      "/" + request->target_id + "/map_matching_action";
    RCLCPP_INFO(this->get_logger(),
                "Initiating a matcher client targeting robot ID %s",
                action_destination.c_str());

    // Initialize action client
    auto new_client_ptr =
      rclcpp_action::create_client<MatchMaps>(this, action_destination);

    if (!new_client_ptr->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
      response->successful = false;
      return;
    } else if (client_ptr_) {
      RCLCPP_ERROR(this->get_logger(), "A client exists already");
      response->successful = false;
      return;
    } else {
      client_ptr_ = new_client_ptr;
      RCLCPP_INFO(this->get_logger(),
                  "Initiated matcher client targeting %s successfully",
                  request->target_id.c_str());
      response->successful = true;
    }

    // Construct and send goal
    auto goal_msg = MatchMaps::Goal();
    ConvertToROS(local_map_, goal_msg.map);

    auto send_goal_options = rclcpp_action::Client<MatchMaps>::SendGoalOptions();
    send_goal_options.goal_response_callback =
      [this](const GoalHandleMatchMaps::SharedPtr& goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
        } else {
          RCLCPP_INFO(this->get_logger(),
                      "Goal accepted by server, waiting for result");
        }
      };

    send_goal_options.feedback_callback =
      [this](GoalHandleMatchMaps::SharedPtr,
             const std::shared_ptr<const MatchMaps::Feedback> feedback) {
        RCLCPP_INFO(this->get_logger(), "Status code received: %d", feedback->status);
      };

    send_goal_options.result_callback =
      [this](const GoalHandleMatchMaps::WrappedResult& result) {
        switch (result.code) {
          case rclcpp_action::ResultCode::SUCCEEDED:
            break;
          case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
            return;
          case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
            return;
          default:
            RCLCPP_ERROR(this->get_logger(), "Unknown result code");
            return;
        }

        const auto& pose_ros = result.result->map_pose;
        Eigen::Affine3d pose_eigen;
        tf2::fromMsg(pose_ros, pose_eigen);

        Eigen::AngleAxisd axang(pose_eigen.matrix().block<3, 3>(0, 0));
        double angle = axang.angle() * axang.axis()(2);

        RCLCPP_INFO(this->get_logger(),
                    "Result received: x: %.5f, y: %.5f, z: %.5f t: %.5f",
                    pose_ros.position.x,
                    pose_ros.position.y,
                    pose_ros.position.z,
                    angle);

        this->client_ptr_.reset();
        RCLCPP_INFO(this->get_logger(), "Client reset successfully");
      };
    this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
  }

private:
  rclcpp_action::Client<MatchMaps>::SharedPtr client_ptr_;
  rclcpp::Service<TriggerMatching>::SharedPtr srv_ptr_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_subscription_ptr_;
  std::unique_ptr<map_matcher::Consensus> consensus_matcher_;
  std::vector<map_matcher::SlicePtr> local_map_;
};
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_matcher_ros::MatcherClient);
