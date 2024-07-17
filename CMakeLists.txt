cmake_minimum_required(VERSION 3.5)
project(tomographic_map_matching CXX)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(teaserpp REQUIRED)
find_package(gflags REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)  # spdlog depends on fmt, but still

# Library
add_library(${PROJECT_NAME} SHARED
  src/map_matcher_base.cpp
  src/consensus.cpp
  src/orb_teaser.cpp
  src/fpfh_ransac.cpp
  src/fpfh_teaser.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
  ${OpenCV_LIBS}
  spdlog::spdlog
  teaserpp::teaser_registration
  nlohmann_json::nlohmann_json
)
target_include_directories(${PROJECT_NAME} PRIVATE include)

# App
add_executable(${PROJECT_NAME}_app src/batch_processing.cpp)
target_link_libraries(${PROJECT_NAME}_app
  ${PROJECT_NAME}
  spdlog::spdlog
  gflags
  fmt::fmt
)
target_include_directories(${PROJECT_NAME}_app PRIVATE include)
