FROM ubuntu:22.04

# Switch the user within docker for file ownership compatibility
ENV USERNAME=user
ARG USER_ID=1000

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    sudo \
    git \
    nano \
    wget \
    tzdata \
    && apt-get clean

RUN useradd -U --uid ${USER_ID} -ms /bin/bash $USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd \
    && adduser $USERNAME sudo \
    && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

# GUI through docker
RUN sudo apt-get install -y --no-install-recommends \
    mesa-utils \
    && sudo apt-get clean

# Build essentials
RUN sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && sudo apt-get clean

# PCL
RUN sudo apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libeigen3-dev \
    libflann-dev \
    libvtk9-dev \
    libvtk9-qt-dev \
    && sudo apt-get clean

ARG PCL_VERSION=1.14.0
RUN wget -q https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-$PCL_VERSION.tar.gz \
    && tar -xf pcl-$PCL_VERSION.tar.gz && rm pcl-$PCL_VERSION.tar.gz

RUN cd pcl-pcl-$PCL_VERSION && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && sudo make -j8 install

# OpenCV
ARG OPENCV_VERSION=4.9.0
RUN wget -q -O opencv.tar.gz https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VERSION.tar.gz \
    && wget -q -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/$OPENCV_VERSION.tar.gz \
    && tar -xf opencv.tar.gz \
    && tar -xf opencv_contrib.tar.gz \
    && rm opencv.tar.gz opencv_contrib.tar.gz

COPY --chown=$USERNAME:$USERNAME misc/estimateRigid2D.patch ./

RUN cd opencv-$OPENCV_VERSION \
    && git apply ../estimateRigid2D.patch \
    && mkdir build && cd build \
    && cmake \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-$OPENCV_VERSION/modules \
    -DBUILD_LIST=calib3d,xfeatures2d,highgui \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_JAVA=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_V4L=OFF \
    -DWITH_FFMPEG=OFF \
    -DVIDEOIO_ENABLE_PLUGINS=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_OPENMP=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH_GTK=OFF \
    -DWITH_QT=ON \
    -DCMAKE_BUILD_TYPE=Release .. \
    && sudo make -j8 install

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
    && sudo make -j8 install

# Map Matching Library & other tools
RUN sudo apt-get install -y --no-install-recommends \
    libspdlog-dev \
    libgflags-dev \
    nlohmann-json3-dev \
    clangd \
    && sudo apt-get clean

RUN sudo install -d -o $USERNAME -g $USERNAME /data
