#!/bin/bash 

topics_to_record="/tf /zed2/zed_node/rgb/image_rect_color /zed2/zed_node/rgb/camera_info /zed2/zed_node/depth/depth_registered /zed2/zed_node/depth/depth_registered/camera_info /zed2/zed_node/odom /zed2/zed_node/pose /imu/data /imu/nav_sat_fix /husky_velocity_controller/odom /zed2/zed_node/imu/data /zed2/zed_node/imu/mag /state_of_charge /odometry/filtered/ /odometry/filtered/local /husky_velocity_controller/cmd_vel /joy_teleop/cmd_vel /joy_teleop/joy /joy_teleop/joy/set_feedback"

#topics_array=($topics_to_record) 

missing_topic=$(echo "${topics_to_record}" | tr " " "\n" | grep -v $(echo "-e $(rostopic list | tr '\n' ' ' | sed -e 's/ / -e /g' -e 's/\(.*\)-e/\1 /')"))
#echo $missing_topic 
#echo "${topics_to_recod}" | tr " " "\n" #| grep -v $(echo "-e $(rostopic list | tr '\n' ' ' | sed -e 's/ / -e /g' -e 's/\(.*\)-e/\1 /')") 
if [ ! -z "$missing_topic" ]; then   
  echo " "
  echo -e "\033[0;33mWarning: next topics are not publish\033[0m"
  echo -e "\033[0;33m${missing_topic}\033[0m"   
  echo " "   
  exit 
fi 
get_file_name=$(rosbag record --duration=2 -o /media/husky/ssd/rosbag/ ${topics_to_record} | grep Recording | sed -e "s/.*\ '//" -e "s/'.//" -e "s/\x1B\[[0-9;]\{1,\}[A-Za-z]//g"  ) 

#missing_topic=$(echo "${topics_to_record}" | tr " " "\n" | grep -v $(echo "-e $(rosbag info ${get_file_name} | tail -n +$(rosbag info ${get_file_name} | grep -n topics | cut -f1 -d:) | sed -e 's#.* \(\/\)#\/#g')") 
missing_topic=$(echo "${topics_to_record}" | tr " " "\n" | grep -v $(echo "-e $(echo $(rosbag info ${get_file_name} | tail -n +$(rosbag info ${get_file_name} | grep -n topics | cut -f1 -d:) | sed -e 's#.* \(\/\)#\/#g' -e 's/\(\ \).*//g') | sed 's/\(\ \)/\ -e\ /g')"))
if [ ! -z "$missing_topic" ]; then   
  echo " "   
  echo -e "\033[0;33mWarning: next topics have no msg registable in 2s\033[0m"   
  echo -e "\033[0;33m${missing_topic}\033[0m"   
  echo " "   
  rm $(echo ${get_file_name}) 
  exit 
fi 

rm $(echo ${get_file_name}) 

echo -e "\033[0;32mAll right, let'go\033[0m"
#echo "$(ls /media/agx/ssd/rosbag_yvette/)" | grep -v -e $(ls /media/agx/Transcend/rosbag/barakuda/yvette_null/ | tr '\n' ' ' | sed -e 's/ / -e /g' -e 's/\(.*\)-e/\1 /') | tr "\n" " "
