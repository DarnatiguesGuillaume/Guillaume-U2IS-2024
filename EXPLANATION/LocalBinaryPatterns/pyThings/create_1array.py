from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt
import sklearn.model_selection
import numpy as np
import argparse
import time
import cv2
import sys
import os

#   Create a single array for histograms and a single array for labels
option_set = [
    [
            #   input:          images,     labels,     size of the suffix to cut at the end of the images name (4 for .png, "." + "p" + "n" + "g")
        "../data/set_2/", "../data/sets_Details/set_2_Labels.txt", 10,
            #   output:         prefix,     data name,  labels name,    suffix
        "../data/processed_Data/set_2", "_Data", "_Labels", ".npy",
            #   additional options:    
        0
    ],
    [
            #   input:          images,     labels,     size of the suffix to cut at the end of the images name (4 for .png, "." + "p" + "n" + "g")
        "../data/set_11/", "../data/sets_Details/set_11_Labels.txt", 4,
            #   output:         prefix,     data name,  labels name,    suffix
        "../data/processed_Data/set_11", "_Data", "_Labels", ".npy",
            #   additional options:     
        0 
    ]
]


#=========================================== OTHERS | NOT PERMANENT =============================================================

def extract_radial_histogram_features(image, num_bins=10): #73% !!!!!
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the FFT of the image and shift the zero frequency component to the center
    fshift = np.fft.fftshift(np.fft.fft2(image))
    magnitude_spectrum = np.abs(fshift)
    
    rows, cols = magnitude_spectrum.shape
    center = (cols // 2, rows // 2)
    max_radius = int(np.sqrt(center[0]**2 + center[1]**2))
    
    # Initialize the histogram bins
    histogram = np.zeros(num_bins)
    bin_edges = np.linspace(0, max_radius, num_bins + 1)

    for y in range(rows):
        for x in range(cols):
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            bin_index = np.digitize(r, bin_edges) - 1
            if bin_index < num_bins:
                histogram[bin_index] += magnitude_spectrum[y, x]

    # Normalize the histogram
    histogram /= histogram.sum()
    
    return histogram

HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

def extract_hog_features(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=False)
    return hog_features

# function for sub images ======================================================

def create_histo(lbp, number_of_bins, sub_images_number):
    grid = np.arange(0, lbp.shape[1]+1, lbp.shape[1]//sub_images_number)
    sub_image_histograms = []
    for i in range(1, len(grid)):
        for j in range(1, len(grid)):
            sub_image = lbp[grid[i-1]:grid[i], grid[j-1]:grid[j]]
            sub_image_histogram = np.histogram(sub_image, bins=number_of_bins)[0]
            sub_image_histograms.append(sub_image_histogram)
    histogram = np.array(sub_image_histograms).flatten()
    return histogram

# function to find in which interval an image is, and the according label ============================

def get_label_for_image(image_number, intervals):
    for start, end, label in intervals:
        if start <= image_number <= end:
            return label
    return None

def calculate_color_proportions(image_array):
    red_channel = image_array[:, :, 0].astype(np.float32)
    green_channel = image_array[:, :, 1].astype(np.float32)
    blue_channel = image_array[:, :, 2].astype(np.float32)
    intensity = red_channel + green_channel + blue_channel
    intensity[intensity == 0] = 1
    red_normalized = red_channel / intensity
    green_normalized = green_channel / intensity
    blue_normalized = blue_channel / intensity
    total_red = np.sum(red_normalized)
    total_green = np.sum(green_normalized)
    total_blue = np.sum(blue_normalized)
    total_sum = total_red + total_green + total_blue
    return np.array([total_red / total_sum, total_green / total_sum, total_blue / total_sum])

# arguments parsing ============================================================================================

parser = argparse.ArgumentParser(description="Process images with LBP and save histograms.")
parser.add_argument("--resize_size", type=int, default=150, help="Size to resize images to.")
parser.add_argument("--radius", type=int, default=1, help="Radius for LBP.")
parser.add_argument("--n_points", type=int, default=8, help="Number of points for LBP.")
parser.add_argument("--number_of_bins", type=int, default=36, help="Number of bins for the histogram.")
parser.add_argument("--sub_images_number", type=int, default=1, help="Number of sub-images.")
parser.add_argument("--method", type=str, choices=["default", "ror", "uniform", "nri_uniform", "var"], default="ror", help="Method for LBP.")
args = parser.parse_args()


# iterates trough datasets ======================================================================================

for Opt in option_set:

    # variable creation =========================================================

    METHOD = args.method
    radius = args.radius
    n_points = args.n_points
    resize_size = args.resize_size
    number_of_bins = args.number_of_bins
    sub_images_number = args.sub_images_number
    # METHOD in ["default", "ror", "uniform", "nri_uniform", "var"]
    vision_Data = []
    vision_Labels = []
    input_Dir = Opt[0]
    label_File = Opt[1]
    intervals = []
    with open(label_File, 'r') as file:
        for line in file:
            start_end, label = line.split(':')
            start, end = start_end.split('-')
            start, end, label = int(start), int(end), int(label.strip())
            intervals.append((start, end, label))
    print(intervals)
    texture_name_list = os.listdir(input_Dir)
    total_number_of_image = len(texture_name_list)
    print(total_number_of_image)

    # print options ================

    bar_size=100
    print("Starting...")
    begin_time = time.time()

    # intervals of images to avoid if needed ==========


# loop =================================================================================================

    i=0
    for name in texture_name_list:
        number = int(name[:-Opt[2]])
        label = get_label_for_image(number, intervals)
        #print(intervals,label,number)
        if False: #normal
            img = cv2.resize(cv2.imread(os.path.join(input_Dir, name), cv2.IMREAD_GRAYSCALE), (resize_size, resize_size))
            lbp = local_binary_pattern(img, n_points, radius, method=METHOD)
            hi = create_histo(lbp, number_of_bins, sub_images_number)
            vision_Data.append(hi)
            vision_Labels.append( label )
        
        if False: #color
            img = cv2.resize(cv2.imread(os.path.join(input_Dir, name), cv2.IMREAD_COLOR), (resize_size, resize_size))
            lbp_channels = [local_binary_pattern(channel, n_points, radius, method='uniform') for channel in cv2.split(img)]
            histograms = [create_histo(lbp, number_of_bins, sub_images_number) for lbp in lbp_channels]
            combined_histogram = np.concatenate(histograms)
            vision_Data.append(combined_histogram)
            vision_Labels.append( s2 )

        if True:
            img_color = cv2.resize(cv2.imread(os.path.join(input_Dir, name), cv2.IMREAD_COLOR), (resize_size, resize_size))
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(img_gray, n_points, radius, method=METHOD)
            histo = create_histo(lbp, number_of_bins, sub_images_number)
            #lbp2 = local_binary_pattern(img_gray, 8, 2, method=METHOD)
            #histo2 = create_histo(lbp, number_of_bins, sub_images_number)
            #histo = np.concatenate((histo, histo2))
            color_proportions = calculate_color_proportions(img_color)
            histo = np.concatenate((histo, color_proportions))


#testestestestestestestest
            #features = extract_radial_histogram_features(img_gray)
            #features2 = extract_hog_features(img_gray)

            #histo = np.concatenate((histo, features))
            #histo = np.concatenate((histo, features2))
#testestestestestestestest

            vision_Data.append(histo)
            vision_Labels.append(label)
            
        
    # print ==================================================

        prop = (i+1)/total_number_of_image
        s = int(bar_size*prop)
        sys.stdout.write(f'\r{(i+1):>6}/{total_number_of_image:<6}   |   {prop*100:.2f}%   [{'#'*s + ' ' * (bar_size - s)}]    Processed {name:<10}')
        sys.stdout.flush()
        i+=1

    # end of loop ====================================================================================

    vision_Data = np.array(vision_Data)
    vision_Labels = np.array(vision_Labels)
    print("\nImages shape: ", vision_Data.shape," and label shape ",vision_Labels.shape)
    np.save(    Opt[3] + Opt[4] + Opt[6]    ,vision_Data)
    np.save(    Opt[3] + Opt[5] + Opt[6]    ,vision_Labels)
    end_time = time.time()
    print("\n\t...Done in ", end_time - begin_time, " seconds !!!")

    # end of data creation ===

if True:
    with open("real_results.txt", 'a') as file:
        file.write('\n' * 3)
        file.write(f'--resize_size {args.resize_size} --radius {args.radius} --n_points {args.n_points} --number_of_bins {args.number_of_bins} --sub_images_number {args.sub_images_number} --method {args.method}\n')
