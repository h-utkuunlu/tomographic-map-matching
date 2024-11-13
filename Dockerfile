FROM cair_unitree_main AS base

# TEASER++
RUN git clone --depth 1 https://github.com/MIT-SPARK/TEASER-plusplus \
    && cd TEASER-plusplus \
    && mkdir build && cd build \
    && cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOC=OFF \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_WITH_MARCH_NATIVE=ON .. \
    && make -j8 install \
    && cd /ros_ws \
    && rm -rf TEASER-plusplus

RUN mkdir -p /ros_ws/src/tomographic

# Separate bringup for faster updates on launch files
COPY cpp /ros_ws/src/tomographic/cpp
COPY ros2/map_matcher_interfaces /ros_ws/src/tomographic/ros2/map_matcher_interfaces
COPY ros2/map_matcher /ros_ws/src/tomographic/ros2/map_matcher
RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build

COPY ros2/map_matcher_bringup /ros_ws/src/tomographic/ros2/map_matcher_bringup
RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --packages-select map_matcher_bringup
