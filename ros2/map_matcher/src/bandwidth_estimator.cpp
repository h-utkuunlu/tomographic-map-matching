#include "conversions.hpp"

#include <filesystem>
#include <map_matcher_interfaces/msg/slice_map.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_compression/compression_options.hpp>
#include <rosbag2_compression/sequential_compression_writer.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace map_matcher_ros {

class BandwidthEstimator : public rclcpp::Node
{
public:
  explicit BandwidthEstimator(const rclcpp::NodeOptions& options)
    : rclcpp::Node("map_matcher_bandwidth_estimator", options)
  {

    this->declare_parameter("parameter_config", "");
    this->declare_parameter("data_config", "");

    std::filesystem::path param_config_path(
      this->get_parameter("parameter_config").as_string()),
      data_config_path(this->get_parameter("data_config").as_string());

    if (not(std::filesystem::exists(param_config_path) and
            param_config_path.extension().string() == ".json")) {
      RCLCPP_ERROR(this->get_logger(),
                   "Invalid parameter config file (%s)",
                   param_config_path.c_str());
      exit(-1);
    }

    if (not(std::filesystem::exists(data_config_path) and
            data_config_path.extension().string() == ".json")) {
      RCLCPP_ERROR(
        this->get_logger(), "Invalid data config file (%s)", data_config_path.c_str());
      exit(-1);
    }

    // Set up matcher
    map_matcher::json parameter_config;
    {
      std::ifstream parameter_config_file(param_config_path.string());
      parameter_config = map_matcher::json::parse(parameter_config_file);
    }
    consensus_matcher_ = std::make_unique<map_matcher::Consensus>(parameter_config);

    // Obtain timestamp for folder naming
    std::string time_string;
    {
      auto now = std::time(nullptr);
      auto now_manip = *std::localtime(&now);
      std::stringstream time_string_stream;
      time_string_stream << std::put_time(&now_manip, "%Y-%m-%d-%H-%M-%S");
      time_string = time_string_stream.str();
    }

    if (!std::filesystem::exists(time_string)) {
      std::filesystem::create_directory(time_string);
    }

    // Bags to write
    {
      // Slice map, uncompressed
      writer_slice_ = std::make_unique<rosbag2_cpp::Writer>();

      rosbag2_storage::StorageOptions storage_options;
      storage_options.uri = time_string + "/slice";
      storage_options.storage_id = "sqlite3";
      writer_slice_->open(storage_options);

      writer_slice_->create_topic({
        0u,
        "sliced_map",
        "map_matcher_interfaces/msg/SliceMap",
        rmw_get_serialization_format(),
        {},
        "",
      });
    }

    {
      // Slice map, compressed
      rosbag2_compression::CompressionOptions compression_options;
      compression_options.compression_mode =
        rosbag2_compression::CompressionMode::MESSAGE;
      compression_options.compression_format = "zstd";

      writer_slice_compressed_ = std::make_unique<rosbag2_cpp::Writer>(
        std::make_unique<rosbag2_compression::SequentialCompressionWriter>(
          compression_options));

      rosbag2_storage::StorageOptions storage_options;
      storage_options.uri = time_string + "/slice-compressed";
      storage_options.storage_id = "sqlite3";
      writer_slice_compressed_->open(storage_options);

      writer_slice_compressed_->create_topic({
        0u,
        "sliced_map",
        "map_matcher_interfaces/msg/SliceMap",
        rmw_get_serialization_format(),
        {},
        "",
      });
    }

    {
      // PCD map, uncompressed
      writer_pcd_ = std::make_unique<rosbag2_cpp::Writer>();

      rosbag2_storage::StorageOptions storage_options;
      storage_options.uri = time_string + "/pcd";
      storage_options.storage_id = "sqlite3";
      writer_pcd_->open(storage_options);

      writer_pcd_->create_topic({
        0u,
        "pcd_map",
        "sensor_msgs/msg/PointCloud2",
        rmw_get_serialization_format(),
        {},
        "",
      });
    }

    {
      // PCD map, compressed
      rosbag2_compression::CompressionOptions compression_options;
      compression_options.compression_mode =
        rosbag2_compression::CompressionMode::MESSAGE;
      compression_options.compression_format = "zstd";

      writer_pcd_compressed_ = std::make_unique<rosbag2_cpp::Writer>(
        std::make_unique<rosbag2_compression::SequentialCompressionWriter>(
          compression_options));

      rosbag2_storage::StorageOptions storage_options;
      storage_options.uri = time_string + "/pcd-compressed";
      storage_options.storage_id = "sqlite3";
      writer_pcd_compressed_->open(storage_options);

      writer_pcd_compressed_->create_topic({
        0u,
        "pcd_map",
        "sensor_msgs/msg/PointCloud2",
        rmw_get_serialization_format(),
        {},
        "",
      });
    }

    // Start processing
    Process();

    RCLCPP_WARN(this->get_logger(), "Done! Exiting...");
    exit(0);
  }

  void Process()
  {

    // Extract data config
    map_matcher::json data_config;
    {
      std::ifstream data_config_file(this->get_parameter("data_config").as_string());
      data_config = map_matcher::json::parse(data_config_file);
    }

    std::filesystem::path data_root_path(data_config["root"]);

    RCLCPP_INFO(this->get_logger(),
                "Data root: %s. Num. pairs: %lu",
                data_root_path.string().c_str(),
                data_config["pairs"].size());

    rclcpp::Clock clock;

    for (const auto& pair : data_config["pairs"]) {

      // Only need to load the first map
      std::filesystem::path pcd_path =
        data_root_path / std::filesystem::path(pair.at(0));

      RCLCPP_INFO(this->get_logger(), "Processing %s", pcd_path.string().c_str());
      rclcpp::Time stamp = clock.now();

      map_matcher::PointCloud::Ptr pcd(new map_matcher::PointCloud());
      pcl::io::loadPCDFile(pcd_path.string(), *pcd);

      // Send to pcd message
      sensor_msgs::msg::PointCloud2 pcd_msg;
      pcl::toROSMsg(*pcd, pcd_msg);

      writer_pcd_->write(pcd_msg, "pcd_map", stamp);
      writer_pcd_compressed_->write(pcd_msg, "pcd_map", stamp);

      std::vector<map_matcher::SlicePtr> sliced_map =
        consensus_matcher_->ComputeSliceImages(pcd);
      consensus_matcher_->ComputeSliceFeatures(sliced_map);

      map_matcher_interfaces::msg::SliceMap map_msg;
      ConvertToROS(sliced_map, map_msg);
      writer_slice_->write(map_msg, "sliced_map", stamp);
      writer_slice_compressed_->write(map_msg, "sliced_map", stamp);
    }

    writer_pcd_->close();
    writer_pcd_compressed_->close();
    writer_slice_->close();
    writer_slice_compressed_->close();
  }

private:
  std::unique_ptr<map_matcher::Consensus> consensus_matcher_;
  std::unique_ptr<rosbag2_cpp::Writer> writer_pcd_, writer_pcd_compressed_,
    writer_slice_, writer_slice_compressed_;
};
} // namespace map_matcher_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(map_matcher_ros::BandwidthEstimator);
