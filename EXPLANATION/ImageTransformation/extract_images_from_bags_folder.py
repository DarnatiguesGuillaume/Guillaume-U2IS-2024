import argparse
import numpy as np
import torch
import rosbag
import os
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bag_dir', type=str, required=True, help='Path to the dir with bag files to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    
    args = parser.parse_args()
    print (1)

    bag_fps = os.listdir(args.bag_dir)
    print('Found {} bag files'.format(len(bag_fps)))

    for i, bag_fp in enumerate(bag_fps):
        print('Converting {}, {}/{}'.format(bag_fp, i+1, len(bag_fps)))

        bag = rosbag.Bag(os.path.join(args.bag_dir, bag_fp))

        # Initialize a counter to keep track of the number of RGB images found
        rgb_count = 0
        
        save_dir = os.path.join(args.save_to, bag_fp[:-4])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        for topic, msg, t in bag.read_messages(topics=["/multisense/left/image_rect_color"]):
            # Convert ROS Image message to numpy array
            np_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

            # Convert numpy array to OpenCV image format (BGR)
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            # Save the image as PNG
            image_save_path = os.path.join(save_dir, f'{bag_fp[:-4]}_rgb_{rgb_count}.png')
            cv2.imwrite(image_save_path, cv_image)

            print('Saved RGB image:', image_save_path)

            # Increment the counter
            rgb_count += 1

        # Check if any RGB images were found in the bag file
        if rgb_count == 0:
            print('No RGB images found in', bag_fp)

        # Close the bag file
        bag.close()

#'/zed_node/depth/depth_registered'