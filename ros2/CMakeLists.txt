cmake_minimum_required(VERSION 3.8)
project(map_matching_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tomographic_map_matching/
  ${CMAKE_CURRENT_BINARY_DIR}/tomographic_map_matching)

find_package(ament_cmake REQUIRED)
find_package(rclcpp_components REQUIRED)

add_library(${CMAKE_PROJECT_NAME}_component SHARED src/map_matcher_node.cpp)
target_compile_features(${CMAKE_PROJECT_NAME}_component PUBLIC cxx_std_20)
target_include_directories(
  ${CMAKE_PROJECT_NAME}_component
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(${CMAKE_PROJECT_NAME}_component tomographic_map_matching)

ament_target_dependencies(
  ${CMAKE_PROJECT_NAME}_component
  rclcpp
  rclcpp_components)

rclcpp_components_register_node(${CMAKE_PROJECT_NAME}_component PLUGIN "tomographic_map_matcher_node::MatcherNode" EXECUTABLE map_matcher_node)

install(TARGETS ${CMAKE_PROJECT_NAME}_component LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()
