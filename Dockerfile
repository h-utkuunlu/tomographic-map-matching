FROM ros:jazzy-perception

# Ubuntu adds a default 1000 user. Use it as the user within container with sudo
# privileges
ARG USERNAME=ubuntu
RUN echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME
USER $USERNAME

# Update repos once
RUN sudo apt-get update

# Mounting point for data
RUN sudo install -d -o $USERNAME -g $USERNAME /data

# Working folder
RUN mkdir -p /home/$USERNAME/ros_ws/src
WORKDIR /home/$USERNAME

# GUI through docker & other dev tools
RUN DEBIAN_FRONTEND=noninteractive sudo apt-get install -y --no-install-recommends \
    mesa-utils \
    nano \
    zsh \
    && sudo apt-get clean

# PCL development files
RUN DEBIAN_FRONTEND=noninteractive sudo apt-get install -y --no-install-recommends \
    libpcl-dev \
    && sudo apt-get clean

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
    && sudo make -j8 install \
    && cd /home/$USERNAME \
    && rm -rf TEASER-plusplus

# Map Matching Library Dependencies
RUN sudo apt-get install -y --no-install-recommends \
    libspdlog-dev \
    libgflags-dev \
    nlohmann-json3-dev \
    && sudo apt-get clean

WORKDIR /home/$USERNAME/ros_ws
