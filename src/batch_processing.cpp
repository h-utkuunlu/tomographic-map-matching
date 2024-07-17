#include <exception>
#include <filesystem>
#include <fmt/core.h>
#include <gflags/gflags.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <tomographic_map_matching/consensus.hpp>
#include <tomographic_map_matching/fpfh_ransac.hpp>
#include <tomographic_map_matching/fpfh_teaser.hpp>
#include <tomographic_map_matching/map_matcher_base.hpp>
#include <tomographic_map_matching/orb_teaser.hpp>

using json = map_matcher::json;

// Flags
DEFINE_string(parameter_config, "",
              "JSON file with parameters. Results are appended");
DEFINE_string(data_config, "",
              "Scenario file that delineates pairs to be tested");
DEFINE_bool(visualize, false, "Enable visualizations");
DEFINE_bool(debug, false, "Enable debugging logs");

Eigen::Matrix4d ReadGTPose(std::string fname) {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  try {
    std::string line;
    std::fstream poseFile;
    poseFile.open(fname, std::ios::in);
    getline(poseFile, line);
    poseFile.close();

    double val;
    int itemCount = 0, row = 0, col = 0;

    while (itemCount++ < 12) { // Each row stores 3x4 tf matrix entries

      val = std::stod(line.substr(0, line.find_first_of(" ")));
      pose(row, col++) = val;

      // Prepare for the next
      line = line.substr(line.find_first_of(" ") + 1);

      // Move to next row if all cols are filled
      if (col == 4) {
        ++row;
        col = 0;
      }
    }
  } catch (std::exception &ex) {
    spdlog::warn("File {} does not exist. Assuming identity pose", fname);
  }

  return pose;
}

// map_matcher::PointCloud::Ptr
// ApplyRandomRollPitch(const map_matcher::PointCloud::Ptr &pcd,
//                      Eigen::Matrix4d pose, double magnitude_deg) {
//   map_matcher::PointCloud::Ptr perturbed_pcd(new map_matcher::PointCloud());
//   Eigen::Matrix4d posei = pose.inverse();
//   pcl::transformPointCloud(*pcd, *perturbed_pcd, posei);

//   std::random_device rd{};
//   std::mt19937 rgen{rd()};

//   // Sample radian, even though degree is specified
//   std::normal_distribution<double> sampler{0.0, magnitude * M_PI / 180.0};

//   double roll = sampler(rgen), pitch = sampler(rgen);
//   spdlog::info("[RP Pert] Roll(rad): {} Pitch(rad): {}", roll, pitch);

//   Eigen::Matrix4d perturbed_pose = Eigen::Matrix4d::Identity();
//   Eigen::AngleAxisd roll_mat(roll, Eigen::Vector3d::UnitX()),
//       pitch_mat(pitch, Eigen::Vector3d::UnitY());

//   perturbed_pose.topLeftCorner(3, 3) = pitch_mat.matrix() *
//   roll_mat.matrix();

//   perturbed_pose.matrix() = pose * perturbed_pose.matrix();
//   pcl::transformPointCloud(*perturbed_pcd, *perturbed_pcd, perturbed_pose);
//   return perturbed_pcd;
// }

double ComputeAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est) {
  return std::abs(std::acos(
      fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)));
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  // Logger config
  if (FLAGS_debug)
    spdlog::set_level(spdlog::level::debug);

  spdlog::set_pattern("[%^%l%$] | %v");

  // Time of execution
  std::string time_string;
  {
    auto now = std::time(nullptr);
    auto now_manip = *std::localtime(&now);
    std::stringstream time_string_stream;
    time_string_stream << std::put_time(&now_manip, "%Y-%m-%d-%H-%M-%S");
    time_string = time_string_stream.str();
  }
  spdlog::info("Start Time: {}", time_string);

  // Ensure the input is somewhat sane
  std::filesystem::path parameter_config_file_path(FLAGS_parameter_config),
      data_config_file_path(FLAGS_data_config);

  if (not(std::filesystem::exists(parameter_config_file_path) and
          parameter_config_file_path.extension().string() == ".json")) {
    spdlog::critical("Parameter config file '{}' is either nonexistent or "
                     "not a .JSON file",
                     parameter_config_file_path.string());
    exit(-1);
  }
  spdlog::info("Parameter config file path: {}",
               parameter_config_file_path.string());

  if (not(std::filesystem::exists(data_config_file_path) and
          data_config_file_path.extension().string() == ".json")) {
    spdlog::critical(
        "Data config file '{}' is either nonexistent or not a .JSON file",
        data_config_file_path.string());
    exit(-1);
  }
  spdlog::info("Data config file path: {}", data_config_file_path.string());

  // Read config file and determine algorithm to be used
  json parameter_config;
  {
    std::ifstream parameter_config_file(parameter_config_file_path.string());
    parameter_config = json::parse(parameter_config_file);
  }

  map_matcher::Parameters base_parameters;
  json parameters_json = base_parameters;

  for (auto &[key, value] : parameter_config.items()) {
    parameters_json[key] = value;
  }

  // TODO: Perhaps a better structure to initialize from a single class?
  std::unique_ptr<map_matcher::MapMatcherBase> matcher;
  if (parameters_json["algorithm"] == 0) { // Consensus
    auto parameters =
        parameters_json.template get<map_matcher::ConsensusParameters>();
    matcher = std::make_unique<map_matcher::Consensus>(parameters);
  } else if (parameters_json["algorithm"] == 1) {
    auto parameters =
        parameters_json.template get<map_matcher::ORBTEASERParameters>();
    matcher = std::make_unique<map_matcher::ORBTEASER>(parameters);
  } else if (parameters_json["algorithm"] == 2) {
    auto parameters =
        parameters_json.template get<map_matcher::FPFHRANSACParameters>();
    matcher = std::make_unique<map_matcher::FPFHRANSAC>(parameters);
  } else if (parameters_json["algorithm"] == 3) {
    auto parameters =
        parameters_json.template get<map_matcher::FPFHTEASERParameters>();
    matcher = std::make_unique<map_matcher::FPFHTEASER>(parameters);
  } else {
    auto algorithm_idx = parameters_json["algorithm"].template get<int>();
    spdlog::critical("Algorithm {} is not implemented", algorithm_idx);
    exit(-1);
  }

  // Pretty-print parameters
  std::stringstream params_str;
  params_str << matcher->GetParameters().dump(2);
  spdlog::info("Parameters: {}", params_str.str());

  spdlog::info("Algorithm: {}", matcher->GetName());

  // Check data file
  json data_config;
  {
    std::ifstream data_config_file(FLAGS_data_config);
    data_config = json::parse(data_config_file);
  }

  // .template get<std::string>()
  std::filesystem::path data_root_path(data_config["root"]);
  spdlog::info("Data root: '{}'. # pairs: {}", data_root_path.string(),
               data_config["pairs"].size());

  // Output file
  std::filesystem::path output_file_folder("/results");
  std::filesystem::path output_file_path =
      output_file_folder /
      std::filesystem::path(matcher->GetName() + "-" + time_string + ".json");

  if (!std::filesystem::exists(output_file_folder)) {
    std::filesystem::create_directory(output_file_folder);
  }
  spdlog::info("Output file: '{}'", output_file_path.string());

  // Start constructing output data
  json output_data;
  output_data["time"] = time_string;
  output_data["data_config"] = data_config_file_path.string();
  output_data["changes"] = parameter_config;
  output_data["full_configuration"] = parameters_json;
  output_data["results"] = json::array();

  spdlog::info("Start processing...");
  spdlog::info("");

  // Process the data with given config
  for (const auto &pair : data_config["pairs"]) {

    std::filesystem::path pcd1_path = data_root_path /
                                      std::filesystem::path(pair.at(0)),
                          pcd2_path = data_root_path /
                                      std::filesystem::path(pair.at(1));
    json stats;
    stats["pcd1"] = pair.at(0);
    stats["pcd2"] = pair.at(1);

    spdlog::info("-> Pair: {} -> {}", pair.at(1), pair.at(0));

    // Load clouds
    map_matcher::PointCloud::Ptr pcd1(new map_matcher::PointCloud()),
        pcd2(new map_matcher::PointCloud());
    pcl::io::loadPCDFile(pcd1_path.string(), *pcd1);
    pcl::io::loadPCDFile(pcd2_path.string(), *pcd2);

    if (pcd1->size() == 0 or pcd2->size() == 0) {
      spdlog::critical("Pointcloud(s) are empty. Aborting");
      output_data["results"].push_back(stats);

      // Record results
      {
        std::ofstream f(output_file_path);
        f << output_data.dump() << std::endl;
      }
      continue;
    }

    stats["pcd1_size"] = pcd1->size();
    stats["pcd2_size"] = pcd2->size();

    // Calculate ground truth poses
    Eigen::Matrix4d pose1 = ReadGTPose(pcd1_path.string().substr(
                                           0, pcd1_path.string().size() - 4) +
                                       "-gtpose.txt"),
                    pose2 = ReadGTPose(pcd2_path.string().substr(
                                           0, pcd2_path.string().size() - 4) +
                                       "-gtpose.txt");

    // // TODO: Apply random roll-pitch if specified
    // map1_pcd = ApplyRandomRollPitch(map1_pcd, pose1);
    // map2_pcd = ApplyRandomRollPitch(map2_pcd, pose2);

    // Determine target pose based on gt info
    Eigen::Matrix4d target = pose1 * pose2.inverse();

    {
      Eigen::Matrix3d rotm = target.block<3, 3>(0, 0);
      Eigen::AngleAxisd axang(rotm);
      double angle = axang.angle() * axang.axis()(2);
      spdlog::info("Target x: {: .5f} y: {: .5f}: z: {: .5f} t: {: .5f}",
                   target(0, 3), target(1, 3), target(2, 3), angle);
      stats["target_x"] = target(0, 3);
      stats["target_y"] = target(1, 3);
      stats["target_z"] = target(2, 3);
      stats["target_t"] = angle;
    }

    // Calculate & print errors
    map_matcher::HypothesisPtr result =
        matcher->RegisterPointCloudMaps(pcd1, pcd2, stats);

    if (result->n_inliers == 0) {
      spdlog::error("Map matching unsuccessful");
    } else {
      stats["result_x"] = result->x;
      stats["result_y"] = result->y;
      stats["result_z"] = result->z;
      stats["result_t"] = result->theta;

      spdlog::info("Result x: {: .5f} y: {: .5f}: z: {: .5f} t: {: .5f}",
                   result->x, result->y, result->z, result->theta);

      double error_position =
          (target.topRightCorner(3, 1) - result->pose.topRightCorner(3, 1))
              .norm();
      double error_angle = ComputeAngularError(
          target.topLeftCorner(3, 3), result->pose.topLeftCorner(3, 3));

      stats["error_position"] = error_position;
      stats["error_angle"] = error_angle;
      double total_time = stats["t_total"].template get<double>();

      std::string log_output =
          fmt::format("Error: {:.5f}m / {:.5f}rad. Took {:.5f}s",
                      error_position, error_angle, total_time);

      if (error_position >
              5.0 * parameters_json["grid_size"].template get<double>() or
          error_angle > 0.1745)
        spdlog::warn("{}", log_output);
      else
        spdlog::info("{}", log_output);
    }

    spdlog::info("");

    output_data["results"].push_back(stats);

    // Record results
    {
      std::ofstream f(output_file_path);
      f << std::setw(2) << output_data << std::endl;
    }

    // Visualization
    if (FLAGS_visualize) {
      matcher->VisualizeHypothesis(pcd1, pcd2, result);
    }
  }

  spdlog::info("Completed!");

  return 0;
}
