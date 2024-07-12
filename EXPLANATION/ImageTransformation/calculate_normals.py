import numpy as np
import argparse
import time
import cv2
import sys
import os






if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--bag_dir', type=str, required=True, help='Folder to take data from')
    parser.add_argument('--save_to', type=str, required=True, help='Folder to save data to')
    parser.add_argument('--sub_fold', action='store_true', help='Option to take inputs from sub-folders')
    
    args = parser.parse_args()

    input_dir = args.bag_dir
    output_dir = args.save_to
    #end---------------------------------------------------------------------------------------------------------------

    # count total number of files 
    total = 0
    count = 0
    def count_files(directory):
        return len([name for name in os.listdir(directory) if (os.path.join(directory,name)).endswith('.npy')])
    #end----------------------------------------------------------------------------------------------------------------

    # function that process each file, this is the main part of this program
    def operations(input,output,co,to):
        for file in os.listdir(input):
            if file.endswith('.npy'):

                image = np.load(os.path.join(input,file))
                dz_dv, dz_du = np.gradient(image)

                # Compute normal components
                Nx = - dz_du / np.sqrt(1 + dz_du**2 + dz_dv**2)
                Ny = - dz_dv / np.sqrt(1 + dz_du**2 + dz_dv**2)
                Nz = 1 / np.sqrt(1 + dz_du**2 + dz_dv**2)
                                    
                def scale(vec):
                    return  abs(vec)*255             # old formula (((aaa + 1) / 2) * 255)
                scaled_Nx = scale(Nx)
                scaled_Ny = scale(Ny)
                scaled_Nz = np.where(image!=0,Nz*255,0)

                #Nz_temp = np.where(image<128,2*Nz*image,Nz*255)
                
                res = np.dstack((scaled_Nx, scaled_Ny, scaled_Nz))
                
                                                                                      # Customize the output name here !
                output_path = os.path.join(output, file[:-14] + "n_original.png") 
                cv2.imwrite(output_path, res.astype(np.uint8))
                #np.save(output_path[:-4]  + ".npy"                     ,res)
                # si on a des problème de place et qu'on veut supprimer les tableaux numpy...
                #os.remove(os.path.join(input,file))

                # Affichage
                co+=1
                percentage = co / to * 100
                progress_str = f"\r{co: <6}/{to: >6} | {round(percentage, 1): >6}%      [{ '#' * int(co / to * 150): <150}]"
                sys.stdout.write(progress_str)
                sys.stdout.flush()
        return co

    #end----------------------------------------------------------------------------------------------------------------

    # this part contains two sub parts, one for processing the .npy inside a folder, the other to explore into that folder  
    #                                                                               folder's and then process the images, depending on the settings chosen
                    # exploring sub folders
            # counting
    if args.sub_fold:

        for dir_name in os.listdir(input_dir):
            input_sub = os.path.join(input_dir,dir_name)
            if os.path.isdir(input_sub):
                total+=count_files(input_sub)
        
            # calcultaing the normals
        print("Starting !")
        start_time = time.time()
        #explore each sub_dir
        for dir_name in os.listdir(input_dir):
            input_sub = os.path.join(input_dir,dir_name)
            if os.path.isdir(input_sub):
                output_sub = os.path.join(output_dir,dir_name)
                if not os.path.exists(output_sub):
                    os.makedirs(output_sub)
                # opening images inside each sub directory
                count = operations(input_sub,output_sub,count,total)
    else:
                    # exploring the root
            # counting
        if os.path.isdir(input_dir):
            total+=count_files(input_dir)
        
            # calcultaing the normals
        print("Starting !")
        start_time = time.time()
        if os.path.isdir(input_dir):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            operations(input_dir,output_dir,count,total)
    #end----------------------------------------------------------------------------------------------------------

    # Time taken
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(" ")
    print(f"Done in : {hours} hours, {minutes} minutes, {seconds} seconds !")
    #end-----------------------------------------------------------------------------------------------------------
