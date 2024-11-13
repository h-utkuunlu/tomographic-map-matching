#include <functional>
#include <memory>
#include <thread>

#include "conversions.hpp"
#include <map_matcher_interfaces/action/match_point_cloud_maps.hpp>
#include <map_matcher_interfaces/action/match_slice_maps.hpp>
#include <map_matcher_interfaces/srv/trigger_matching.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
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
  using MatchSliceMaps = map_matcher_interfaces::action::MatchSliceMaps;
  using GoalHandleMatchSliceMaps = rclcpp_action::ClientGoalHandle<MatchSliceMaps>;

  using MatchPointCloudMaps = map_matcher_interfaces::action::MatchPointCloudMaps;
  using GoalHandleMatchPointCloudMaps =
    rclcpp_action::ClientGoalHandle<MatchPointCloudMaps>;

  using TriggerMatching = map_matcher_interfaces::srv::TriggerMatching;

  explicit MatcherClient(const rclcpp::NodeOptions& options)
    : rclcpp::Node("map_matcher_client", options)
  {

    using namespace std::placeholders;

    // Parameters
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
        map_topic, 10, std::bind(&MatcherClient::MapCallback, this, _1));
      grid_subscription_ptr_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        grid_topic, 10, std::bind(&MatcherClient::GridCallback, this, _1));
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

    slice_map_publisher_ =
      this->create_publisher<map_matcher_interfaces::msg::SliceMap>("matched_slice_map",
                                                                    10);
    // pcd_map_publisher_ =
    //   this->create_publisher<sensor_msgs::msg::PointCloud2>("matched_pcd_map", 10);

    matched_map_publisher_ =
      this->create_publisher<nav_msgs::msg::OccupancyGrid>("other_map", 10);

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

  void GridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {

    RCLCPP_INFO(this->get_logger(), "Received new grid map");
    local_grid_ = *msg;
    return;
  }

  void TriggerServiceHandle(const std::shared_ptr<TriggerMatching::Request> request,
                            std::shared_ptr<TriggerMatching::Response> response)
  {
    const std::string action_destination =
      "/" + request->target_id + "/map_matching_action";
    std::string matching_type;

    if (request->type == 0) {
      matching_type = "sliced";
    } else if (request->type == 1) {
      matching_type = "raw cloud";
    }

    RCLCPP_INFO(this->get_logger(),
                "Initiating a matcher client targeting '%s' with matching type '%s'",
                action_destination.c_str(),
                matching_type.c_str());

    // Initialize action client
    auto new_client_ptr =
      rclcpp_action::create_client<MatchSliceMaps>(this, action_destination);

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
    auto goal_msg = MatchSliceMaps::Goal();
    ConvertToROS(local_map_, goal_msg.map);
    goal_msg.grid_map = local_grid_;

    slice_map_publisher_->publish(goal_msg.map);

    auto send_goal_options = rclcpp_action::Client<MatchSliceMaps>::SendGoalOptions();
    send_goal_options.goal_response_callback =
      [this](const GoalHandleMatchSliceMaps::SharedPtr& goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
        } else {
          RCLCPP_INFO(this->get_logger(),
                      "Goal accepted by server, waiting for result");
        }
      };

    send_goal_options.feedback_callback =
      [this](GoalHandleMatchSliceMaps::SharedPtr,
             const std::shared_ptr<const MatchSliceMaps::Feedback> feedback) {
        RCLCPP_INFO(this->get_logger(), "Status: '%s'", feedback->status.c_str());
      };

    send_goal_options.result_callback =
      [this](const GoalHandleMatchSliceMaps::WrappedResult& result) {
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
  rclcpp_action::Client<MatchSliceMaps>::SharedPtr client_ptr_;
  rclcpp::Service<TriggerMatching>::SharedPtr srv_ptr_;

  // Map publisher for introspection
  rclcpp::Publisher<map_matcher_interfaces::msg::SliceMap>::SharedPtr
    slice_map_publisher_;
  // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_map_publisher_;

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr matched_map_publisher_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_subscription_ptr_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_subscription_ptr_;
  std::unique_ptr<map_matcher::Consensus> consensus_matcher_;
  std::vector<map_matcher::SlicePtr> local_map_;
  nav_msgs::msg::OccupancyGrid local_grid_;
};

} // namespace map_matcher_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_matcher_ros::MatcherClient);
