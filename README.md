# Guillaume-U2IS-2024
This repository contains my work as an intern at the U2IS Research Lab at ENSTA Paris in 2024. It includes my projects, along with the code and documentation developed during my internship, focusing on texture classification for unsupervised robotic navigation in unstructured terrain using vision and IMU (Inertial Measurement Unit) data.



There are 4 folders that constitute the essential part of my internship:

  - EasySegment, a tool to extract masks from images manually using segment-anything from META to make the process easier, and faster.
                  Location on my computer, with segment anything and the docker installed : /home/guillaume-darnatigues/Documents/ros_noetic_Docker/segment_anything/
  - Husky, every program and tool to setup a Clearpath Husky robot and record rosbags on it.
                  Location on the Husky Paquerette in U2IS's garage, with the currently recorded rosbags on the 12th of july 2024 : /media/husky/ssd/
  - ImageTransformation, pretty elementary tools used too extract images and depth images from rosbags, as well as calculating the normals of an image from a depth image.
  - LocalBinaryPatterns, tools to create ground-classification models, using Local Binary Patterns inputted into SVM, LogReg and KNN algorythms.
                  Location on my computer, with the datasets : /home/guillaume-darnatigues/Documents/code/groundClass_NEW/

    My computer was the one labelled 190147 in room R220.
