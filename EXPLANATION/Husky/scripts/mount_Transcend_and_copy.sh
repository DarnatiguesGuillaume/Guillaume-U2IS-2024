#!/bin/bash

check_var="$(mount | grep Transcend)"
if [ "$check_var" == "" ]; then
  echo "mount Transcend"
  sudo mount /dev/disk/by-label/Transcend /media/husky/Transcend
fi

#sudo cp -r /media/husky/ssd/rosbag/husky/* /media/husky/Transcend/rosbag/husky/dataset_traversability/
rsync --progress -r --ignore-existing /media/husky/ssd/rosbag/2024_husky_rosbags/* /media/husky/Transcend/rosbag/husky/dataset_traversability/

