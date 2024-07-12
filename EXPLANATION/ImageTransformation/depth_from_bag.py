import argparse
import numpy as np
import os
import rosbag
#from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bag_dir', type=str, required=True, help='Path to the dir with bag files to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')

    args = parser.parse_args()

    bag_fps = os.listdir(args.bag_dir)
    print('Found {} bag files'.format(len(bag_fps)))

    for i, bag_fp in enumerate(bag_fps):

        save_path = os.path.join(args.save_to, f'{bag_fp[:-4]}_depth')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        print('Converting {}, {}/{}'.format(bag_fp, i+1, len(bag_fps)))

        bag = rosbag.Bag(os.path.join(args.bag_dir, bag_fp))

        depth_count = 0

        bridge = CvBridge()

        #topic=["/device_0/sensor_0/Depth_0/image/data"]
        topic=['/zed_node/depth/depth_registered']
        for topic, msg, t in bag.read_messages(topics=topic):
            np_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Save the depth data as numpy array
            npy_save_path = os.path.join(save_path, f'{bag_fp[:-4]}_depth_{t}.npy')
            np.save(npy_save_path, np_depth_image)

            print('Saved Depth data:', npy_save_path)

            # Increment the counter
            depth_count += 1

        # Check if any depth images were found in the bag file
        if depth_count == 0:
            print('No Depth images found in', bag_fp)

        # Close the bag file
        print("Saved ",depth_count, " images !")
        bag.close()