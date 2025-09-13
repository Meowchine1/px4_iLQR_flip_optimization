.PHONY: all kill

SHELL := /bin/bash

all:
	colcon build --packages-select px4_offboard
	source /opt/ros/humble/setup.bash ; source install/setup.bash ; ros2 launch px4_offboard offboard_velocity_control.launch.py

kill:
	sudo pkill -f gzserver
	sudo pkill -f gz
	sudo pkill -f px4_sitl
	sudo pkill -f px4
#	sudo pkill -f px4_interaction
#	sudo killall -KILL QGroundControl.AppImage
#	sudo killall -KILL px4
#	sudo killall -KILL gz
#	sudo killall -KILL make
# 	sudo killall -KILL ninja
# 	sudo killall -KILL cmake
# 	sudo killall -KILL MicroXRCEAgent
# 	sudo killall -KILL python3
# 	sudo killall -KILL /usr/bin/python3
# 	sudo killall -KILL gnome-terminal
# 	sudo killall -KILL gnome-terminal-server

