#!/bin/bash

SSD_USAGE=$(df -h | grep '/media/husky/ssd' | awk '{print $5}')
echo "Current SSD (/media/husky/ssd) usage: $SSD_USAGE"

rosbag record --duration=5m -b 0 -o "/media/husky/ssd/rosbag/$(date +"%Y")_husky_rosbags/$(date +"%B" | tr '[:upper:]' '[:lower:]')_rosbags/$(date +"%d")_rosbags/${1}_${2}" \
 /tf \
 /zed2/zed_node/rgb/image_rect_color \
 /zed2/zed_node/rgb/camera_info \
 /zed2/zed_node/depth/depth_registered \
 /zed2/zed_node/depth/depth_registered/camera_info \
 /zed2/zed_node/imu/data \
 /zed2/zed_node/imu/mag \
 /zed2/zed_node/odom  \
 /zed2/zed_node/pose  \
 /imu/data \
 /imu/nav_sat_fix \
 /state_of_charge \
 /odometry/filtered/ \
 /odometry/filtered/local \
 /joy_teleop/cmd_vel \
 /joy_teleop/joy \
 /joy_teleop/joy/set_feedback
 /husky_velocity_controller/odom \
 /husky_velocity_controller/cmd_vel \

SSD_USAGE=$(df -h | grep '/media/husky/ssd' | awk '{print $5}')
echo "Current SSD (/media/husky/ssd) usage: $SSD_USAGE"
