#!/bin/bash

rm ../models_Supervised/*
python3 create_1array.py --resize_size $1 --radius $2 --n_points $3 --number_of_bins $4 --sub_images_number $5 --method $6
python3 Make_all.py
python3 Test_all.py
