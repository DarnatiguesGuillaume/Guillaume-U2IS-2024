#!/bin/bash
rosnode kill /zed_node
rosnode kill /zed2/zed2_state_publisher
rosnode kill /zed2/zed_node
echo "!!! don't forget to change the date !!!"
docker start zed_container
docker exec zed_container bash -c "\
  source /opt/ros/noetic/setup.bash && \
  source /catkin/devel/setup.bash && \
  roslaunch zed_wrapper zed2.launch \
" &
rosnode kill /sbg_device
roslaunch husky_bringup guillaume.launch
