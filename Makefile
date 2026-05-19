IMAGE ?= ros2_crazyflie
CONTAINER ?= ros2_crazyflie
ROS_DOMAIN_ID ?= 132
NEUROMORPHIC_ROOT ?= /home/$(USER)/neuromorphic
INTEL_ROOT ?= /intel
CRAZYFLIE_WS ?= $(NEUROMORPHIC_ROOT)/ros2-crazyflie-mocap/ws
HOST_SSH_DIR ?= /home/$(USER)/.ssh

BRIDGE_ARGS ?=
BRIDGE_BASE_ARGS ?= --ros-args -p admm_nxcore_path:=$(NEUROMORPHIC_ROOT)/admm_nxcore
PLOT_LOG ?=
PLOT_ARGS ?=

.PHONY: build run attach loihi-shell setup-ssh build-ws setup-intel-cflib rebuild-nx-streaming-output smoke smoke-ssh bridge-log bridge-mock bridge-real plot-loihi-log

build:
	docker build . -t $(IMAGE)

run:
	@if command -v xhost >/dev/null 2>&1; then xhost +local:docker; fi
	touch .bash_history
	docker run -it --rm --name $(CONTAINER) \
	--privileged \
	--env="DISPLAY" \
	--env="ROS_DOMAIN_ID=$(ROS_DOMAIN_ID)" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="/home/$(USER)/.Xauthority:/root/.Xauthority" \
	--cap-add=NET_RAW \
	--cap-add=NET_ADMIN \
	--device=/dev/bus/usb \
	--volume="$(INTEL_ROOT):/intel" \
	--volume="$(NEUROMORPHIC_ROOT):$(NEUROMORPHIC_ROOT)" \
	--volume="$(HOST_SSH_DIR):/host_ssh:ro" \
	--volume="./ws:/root/ros2_ws" \
	--volume="./.bash_history:/root/.bash_history" \
	--network host \
	--workdir="$(CRAZYFLIE_WS)" \
	$(IMAGE)

attach:
	docker exec -it $(CONTAINER) /bin/bash

loihi-shell:
	docker exec -it $(CONTAINER) bash -lc 'source /opt/ros/humble/setup.sh && source $(CRAZYFLIE_WS)/install/setup.sh && source /intel/variables.sh eth && source /intel/venv/bin/activate && cd $(CRAZYFLIE_WS) && exec bash --noprofile --norc -i'

setup-ssh:
	docker exec $(CONTAINER) bash -lc 'install -d -m 700 /root/.ssh && cp /host_ssh/id_ed25519 /root/.ssh/id_ed25519 && cp /host_ssh/config /root/.ssh/config && chmod 600 /root/.ssh/id_ed25519 /root/.ssh/config && ssh-keyscan -H 10.42.0.100 >> /root/.ssh/known_hosts && chmod 600 /root/.ssh/known_hosts'

build-ws:
	docker exec $(CONTAINER) bash -lc 'source /opt/ros/humble/setup.sh && cd $(CRAZYFLIE_WS) && colcon build --symlink-install --packages-select crazyflie_bridge'

setup-intel-cflib:
	docker exec $(CONTAINER) bash -lc 'git config --global --add safe.directory $(NEUROMORPHIC_ROOT)/crazyflie-lib-python && source /intel/variables.sh eth && source /intel/venv/bin/activate && python3 -m pip install pyusb libusb-package pyserial packaging && python3 -m pip install --no-deps -e $(NEUROMORPHIC_ROOT)/crazyflie-lib-python'

rebuild-nx-streaming-output:
	docker exec $(CONTAINER) bash -lc 'cd /intel/nxkernel/include/nxcore/nxcore-2.5.19/nxcore/graph/streaming/output && make clean && make'

smoke:
	docker exec $(CONTAINER) bash -lc 'source /opt/ros/humble/setup.sh && source $(CRAZYFLIE_WS)/install/setup.sh && source /intel/variables.sh eth && source /intel/venv/bin/activate && python3 -c "import socket, sys, usb.core, numpy, rclpy, cflib, nxcore, nxkernel; sys.path.insert(0, \"$(NEUROMORPHIC_ROOT)/admm_nxcore\"); from nxcore.arch.n3b.n3board import N3Board; from nxkernel.groups.eth_group import EthernetOutputServer; from crazyflie_bridge.crazyflie_bridge_node import default_admm_nxcore_path; from admm_mpc import nx as admm_nx; print(\"python\", sys.executable); print(\"numpy\", numpy.__version__); print(\"admm_nxcore\", default_admm_nxcore_path()); print(\"admm_nx\", admm_nx.__file__); print(\"loihi_runtime\", admm_nx.has_loihi_runtime()); assert admm_nx.has_loihi_runtime(); s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3)); s.close(); print(\"raw socket ok\"); radios = list(usb.core.find(find_all=True, idVendor=0x1915, idProduct=0x7777)); print(\"crazyradio count:\", len(radios))"'

smoke-ssh:
	docker exec $(CONTAINER) bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=5 10.42.0.100 true && echo "loihi ssh ok"'

bridge-log:
	docker exec -it $(CONTAINER) bash -lc 'source /opt/ros/humble/setup.sh && source $(CRAZYFLIE_WS)/install/setup.sh && source /intel/variables.sh eth && source /intel/venv/bin/activate && cd $(CRAZYFLIE_WS) && python3 -m crazyflie_bridge.crazyflie_bridge_node $(BRIDGE_BASE_ARGS) -p arm_on_connect:=false -p position_commands_enabled:=false -p loihi_backend:=disabled $(BRIDGE_ARGS)'

bridge-mock:
	docker exec -it $(CONTAINER) bash -lc 'source /opt/ros/humble/setup.sh && source $(CRAZYFLIE_WS)/install/setup.sh && source /intel/variables.sh eth && source /intel/venv/bin/activate && cd $(CRAZYFLIE_WS) && python3 -m crazyflie_bridge.crazyflie_bridge_node $(BRIDGE_BASE_ARGS) -p arm_on_connect:=false -p position_commands_enabled:=false -p loihi_backend:=mock $(BRIDGE_ARGS)'

bridge-real:
	docker exec -it $(CONTAINER) bash -lc 'source /opt/ros/humble/setup.sh && source $(CRAZYFLIE_WS)/install/setup.sh && source /intel/variables.sh eth && source /intel/venv/bin/activate && cd $(CRAZYFLIE_WS) && python3 -m crazyflie_bridge.crazyflie_bridge_node $(BRIDGE_BASE_ARGS) -p arm_on_connect:=false -p position_commands_enabled:=false -p loihi_backend:=real $(BRIDGE_ARGS)'

plot-loihi-log:
	docker exec $(CONTAINER) bash -lc 'source /intel/venv/bin/activate && cd $(CRAZYFLIE_WS) && python3 plot_loihi_bridge_log.py $(PLOT_LOG) $(PLOT_ARGS)'
