cmake_minimum_required(VERSION 3.8)
project(map_matcher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(BUILD_APP OFF CACHE INTERNAL "Build batch processing executable")
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/tomographic_map_matching/
  ${CMAKE_CURRENT_BINARY_DIR}/tomographic_map_matching
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(map_matcher_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_eigen REQUIRED)

# Server
add_library(${CMAKE_PROJECT_NAME}_server_component SHARED src/map_matcher_server.cpp)
target_compile_features(${CMAKE_PROJECT_NAME}_server_component PUBLIC cxx_std_20)
target_link_libraries(${CMAKE_PROJECT_NAME}_server_component tomographic_map_matching)

ament_target_dependencies(
  ${CMAKE_PROJECT_NAME}_server_component
  rclcpp
  rclcpp_action
  rclcpp_components
  cv_bridge
  pcl_conversions
  map_matcher_interfaces
  tf2_eigen
)

rclcpp_components_register_node(
  ${CMAKE_PROJECT_NAME}_server_component
  PLUGIN "map_matcher_ros::MatcherServer"
  EXECUTABLE map_matcher_server
)

# Client
add_library(${CMAKE_PROJECT_NAME}_client_component SHARED src/map_matcher_client.cpp)
target_compile_features(${CMAKE_PROJECT_NAME}_client_component PUBLIC cxx_std_20)
target_link_libraries(${CMAKE_PROJECT_NAME}_client_component tomographic_map_matching)

ament_target_dependencies(
  ${CMAKE_PROJECT_NAME}_client_component
  rclcpp
  rclcpp_action
  rclcpp_components
  cv_bridge
  pcl_conversions
  map_matcher_interfaces
  tf2_eigen
)

rclcpp_components_register_node(
  ${CMAKE_PROJECT_NAME}_client_component
  PLUGIN "map_matcher_ros::MatcherClient"
  EXECUTABLE map_matcher_client
)

install(
  TARGETS ${CMAKE_PROJECT_NAME}_server_component ${CMAKE_PROJECT_NAME}_client_component
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
