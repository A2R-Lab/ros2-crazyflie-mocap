FROM osrf/ros:humble-desktop

WORKDIR /root/ros2_ws

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    binutils-multiarch \
    gcc-multilib \
    g++-multilib \
    gcc-riscv64-unknown-elf \
    git \
    git-lfs \
    libgoogle-glog-dev \
    libgrpc++1 \
    libc6-dev-i386 \
    libnewlib-dev \
    libpcap-dev \
    make \
    openssh-client \
    ros-humble-rosbag2-storage-mcap \
    python3-pyqtgraph \
    python3-pip \
    python3-setuptools \
    && apt-get clean autoclean \
    && apt-get autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/ \
    && echo "source /opt/ros/humble/setup.bash; source /root/ros2_ws/install/local_setup.sh" >> /root/.bashrc \
    && /ros_entrypoint.sh

RUN pip3 install cflib cfclient autograd

COPY ./ws/src /root/ros2_ws/src

RUN . /opt/ros/humble/setup.sh && cd /root/ros2_ws && colcon build --symlink-install 
