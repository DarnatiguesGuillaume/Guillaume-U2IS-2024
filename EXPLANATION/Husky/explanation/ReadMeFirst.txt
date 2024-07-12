This is the first file you want to look at if you are trying to navigate /media/husky/ssd/ to record rosbags or use them.
Here is documentation by me, Guillaume Darnatigues about the organisation in /media/husky/ssd/, if my way of organising files and folders doesn't suit you, you are not allowed to complain.

There are 5 folders in ssd/, one for documentation purpose, one for the docker build that the zed2 camera works on, one for useful scripts, one for rosbag recording and storage, and one
that i don't know about, surely a thing of the ssd itself: lost+found/.

The only two you should care about are the one about rosbags, conveniently named rosbag/ by it's thoughtful creator (what a smart and organised man), and the one full of scripts, named scripts/.
In case you want to take rosbags, just read the rosbag/ section, in case you want to record rosbags and so on, you have to read both rosbag/ and scripts/ sections, i swear.
At least the "init_docker_zed_and_sdk.sh" file in scripts.

- If you want to quickly read this, you may want to go through the Usage section of every file and folder, that will be quick and give you a good overview of everything

rosbag/
In rosbag, you can find two folders "2024_husky_rosbags/"  and "husky/"  and three bash scripts: "create_folder_of_the_day.sh", "rosbag_record_guis.sh" and  "verif_rostopics.sh".
	create_folder_of_the_day.sh
		Usage
			- Before the first time you record a bag, everyday you do so.
		Goal
			- This is used to create the folder where the bags you record are stored, as the name is dependent on the day of recording for organisation reasons, this program
				finds the date and create the necessary folders.
		
	verif_rostopics.sh
		Usage
			- Before you record rosbags, every time you restart the robot or change something in the code that could cause a problem (literally everything).
		Goal 
			- This is a gift from Adrien Poiré that permits you to check if all topics you are trying to record are actually published and actives.
		Variables		
			- you can change the list of topics you are trying to record by simply editing the first few lines of the problem, editing the variable "topics_to_record"
			- you can also change the location that the test bag is put in by changing the "get_file_name" variable, even though keep in mind that the bag created to 
				check the topics is temporary, so just change this variable if you change the folder's architecture, so that if points to an existing place.

	rosbag_record_guis.sh
		Usage
			- To record a bag, after create_folder_of_the_day and verif_rostopics, with two arguments for the naming of the bag.
		Goal
			- This is the tool to record rosbags during my time of work on paquerettes, from mai 2024 to July 2024.
		Functioning
			- Note that there is a maximum time limit on the rosbag of 5 min
			- Do remember that starting to record a bag then losing connection will not stop the recording process, you need to reconnect and ctrl+C, or wait until the 5 min have passed.  
			- I don't know what the "-b 0" option does, this is setting the buffer to 0, whatever, it was on the previous recording scripts, i just took it 
		Variables
			- When you call the program, the first two arguments you pass will be used in the name of the bag, named: [argument 1]_[argument 2]_[information about the time of recording].bag
				as explained later, argument 1 is usually the general area where you recorded the bag like "ENSTA, SOCCER,... " in all caps, and argument 2 is usually the ground type
				or something to make you remember where you recorded it "grass, gravel, bigGreavel, road,... ". 
			- You can add or remove topics by adding or deleting lines from lines 4 to the end
			- You can change where the files are saved by changing what is after the "-o", currently, it is
				/media/husky/ssd/rosbag/[year]_husky_rosbags/[month]_rosbags/[day number]_rosbags/[location of the recording]_[ground type], do acknowledge that the name of the bag
				will be [location of the recording]_[ground type]_[information about the time of recording], i don't know how to change the last part, and the first part is still
				pretty free, it is just to be able to quickly remember where i recorded the bag examples of names i used are ENSTA_grass because i recorded in the grass in front
				of the school, or SOCCER_gravel for a bag recorded next to the soccer field, in the gravel. If you try to record a bag in a folder that is not created, you will get an 
				error, you need to create the folders first, so like every month and every new day, this can be automated easily, do it if you want ! 
				I will talk more about rosbag organisation just below.

	"husky/"
		Usage
			- Legacy folder
		Legacy folder for bag storage, in case you need some bags you can take them from here but there are topics missing and bad video quality and frame rate, you can delete it for storage
		i won't get mad don't worry, kind of useless folder.

	"2024_husky_rosbags/"
		Usage
			- To store the rosbags.
		This is the storage space, The names of the sub folders are pretty redundant repeating "rosbags" but i thought this made the reading easier and navigation faster.
		The data is saved in [year of recording]_husky_rosbags/[month of recording]_rosbags/[day of recording]_rosbags/[name of the individual bag].bag
		!!! You have to create the folder path before recording a bag using "create_folder_of_the_day.sh".
		As said earlier too, the name of the bag is currently just a quick way to know where it was recorded, with the first word being something to describe the general area, then "_" then
		the ground type or "road" or something, for you to decide later what category you will put it in. Sometimes i record bag that we do not currently use, named "transition" which is a bag 
		of the robot going thought all the types of ground of a location one after the other, or sometimes "return" where i just conduct the robot to another place and record it, in case
		we need longer videos to try our finalised pipelines on (in the far future...). 
		
		Up to 2024, june the 25th included, the images were recorded in low quality, 360 by 640 or something. After that we record in HD1080 at about 22 to 25 fps because of zed2's behavior.

scripts/
In scripts, you can find a couple useful and even essential scripts, namely "init_docker_zed_and_sdk.sh", "mount_Transcend_and_copy.sh" and "wake_paquerettes_and_update_date.sh".
	init_docker_zed_and_sdk.sh
		Usage
			- Every time you restart the robot, before you record a bunch of bags that involve the zed2 camera or the sdk node.
		Goal
			- Due to issues with the zed not using gpu, we figured some kind of incompatibilities between version where surely at cause, and so decided to put the zed wrapper
				and whatever permits you to launch the nodes on a docker, this program quickly kills the useless existing nodes, starts the docker container, source the docker,
				and launch the zed with config files you can edit by manually launching the container.
			- It also launches the sdk node which wouldn't start for some reason, killing it before in case you reuse the program.
		About config files
			- They are stored somewhere on the docker, go to the catkin workspace, then source, then something, then params,...
			- the files currently used (you can modify it in the zed2.launch file) are common.yaml and zed2.yaml, common.yaml controls pretty much all the parameters and zed2.yaml controls
				a couple others, most importantly the image recording quality, currently set to HD1080 because the legacy footage uses this quality.
	
	mount_Transcend_and_copy.sh
		Usage
			- When you want to transfer the rosbags from the ssd to an external Transcend device.
		Goal
			- Mount the Transcend Hard Drives and copy all not yet downloaded rosbags from 2024_husky_rosbags/ and subfolders to the Transcend drive.
		Variables
			- you can restrict the copying to a subfolder or a single bag by changing what is copied, currently it is "/media/husky/ssd/rosbag/2024_husky_rosbags/*"
			- you can obviously change the destination, currently "/media/husky/Transcend/rosbag/husky/dataset_traversability/", which was the chosen folder before i came here.
	
	wake_paquerettes_and_update_date.sh
		Usage
			- Before you use any of the scripts; init_docker_zed_and_sdk.sh, rosbag_record_guis.sh, create_folder_of_the_day.sh, really anything that needs an updated date.
			REMEMBER that this script is meant to be used on your computer, not in the robot, it is just here for storage purpose. 
		Goal
			- For most programs to work, you need the date to be updated, which is not done by restarting Paquerettes. It is really tedious to do so manually every times, so
				this scripts does it for you, and will save you from anger and maybe tears. This script is meant to be on the computer or laptop you are using to reach the robot
				but is kept there too for safe keeping.
		Variables
			- for the ssh connection, the script needs the remote user and the remote host, for the husky this is REMOTE_USER="husky" and REMOTE_HOST="11.0.0.11".
