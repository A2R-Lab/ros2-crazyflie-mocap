IMAGE_NAME=ros2_crazyflie.sif
INSTANCE_NAME=ros2_crazyflie

build:
	apptainer build $(IMAGE_NAME) ros2_crazyflie.def

run:
	@# Start the background instance (automatically sees your $HOME)
	@apptainer instance start $(IMAGE_NAME) $(INSTANCE_NAME) || true
	apptainer shell instance://$(INSTANCE_NAME)
	@apptainer instance stop $(INSTANCE_NAME)

attach:
	apptainer exec instance://$(INSTANCE_NAME) /bin/bash

stop:
	apptainer instance stop $(INSTANCE_NAME)
