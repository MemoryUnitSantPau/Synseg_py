import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tifffile import imread
from scipy.ndimage import label, generate_binary_structure
from pathlib import Path, PureWindowsPath
import argparse


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def Densiometer(PATH, 
         PROJECT_TECHNIQUE = "STORM", 
         connectivity = 3,
         threshold = 0, 
         x_y_ratio = None, 
         z_ratio = None)-> None:
    """
    Main function to perform density calculations on image files.

    Parameters:
        PATH (str): The path to the directory containing image files.
        PROJECT_TECHNIQUE (str): The project technique (default: "STORM").
        neuropil_channel (str): The neuropil channel identifier (default: "SYPH").
        connectivity (int): The connectivity for object identification (default: 3).
        threshold (float): The threshold value for object identification (default: 0).
        x_y_ratio (float): The pixel size ratio for X and Y dimensions (default: None).
        z_ratio (float): The pixel size ratio for the Z dimension (default: None).

    Returns:
        None
    
    Generates:
        Density results_().csv : This file contains the amount of identifyed objects, and the density  
    """

    # Windows path management
    Wpath = PureWindowsPath(PATH)
    PATH = Path(Wpath)
    # LINUX 
    #PATH = Path(PATH)

    # Generate the required objects based on the user inputs.
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    ## Pixel size ratio selection
    if x_y_ratio == None:
        if PROJECT_TECHNIQUE == "STORM":
            x_y_ratio = 0.008
        elif PROJECT_TECHNIQUE == "ARRAY":
            x_y_ratio = 0.4274 
    
    if z_ratio == None:
        if PROJECT_TECHNIQUE == "STORM":
            z_ratio = 0.07
        elif PROJECT_TECHNIQUE == "ARRAY":
            z_ratio = 0.07
        
    ## Structure for object detection
    if connectivity == 2:
        structure = generate_binary_structure(2,1) # 2 dimention connectivity kernel
    elif connectivity == 3:
         structure = generate_binary_structure(3,1) # 3 dimention connectivity kernel
    else:
        return print(f"Error: Connectivity value not accepted. Select a connectivity of 2 or 3 dimentions.")

    ## List the image files to process
    file_list = [file for file in os.listdir(PATH) if file.endswith(".tif")]

    ## Generate an empty dataframe to store the results
    df = pd.DataFrame()

    # MAIN LOOP
    for filename in file_list: # Iterate through files

        ## Generate empty lists to populate with results
        names = []
        proteins = []
        images = []
        shapes = []
        areatotals = []
        labels = []
        obj_ns = []
        densities = []

        ## Read image
        image = imread(os.path.join(PATH, filename)).astype(np.float32)
        ## Normalize image values
        image = image*(255/np.max(image))
        print(filename)

        ## Select the protein channel name
        protein = filename.split('_')[1]

        frame_n = 1 if len(image.shape) == 2 else image.shape[0]
        x, y = image.shape[-2:] if len(image.shape) == 3 else image.shape[:2]

        names.append(filename)
        proteins.append(protein)
        images.append(image)
        shapes.append([frame_n, x, y])
        areatotals.append(frame_n * x * y)

        # Thresholded object identification
        th_image = np.zeros_like(image)
        object_per_stack = 0

        # Calculate total objects without accounting for continuity between frames (NO USED)
        for i in range(frame_n):
            
            th_image[i,:,:] = image[i,:,:] > np.max(image[i,:,:])*threshold
            layer_objs = label(th_image[i,:,:])[1]
            object_per_stack = object_per_stack + layer_objs

        # Calculate the total objects accounting for continuity between frames.
        th_image = th_image.astype(np.uint8)*255
        labeled_image, object_number = label(th_image, structure)
        
        labels.append(labeled_image)
        obj_ns.append(object_number)

        for i in range(len(names)):
            densities.append(((obj_ns[i]/areatotals[i])/(x_y_ratio*x_y_ratio*z_ratio))*10**9)

        pre_df = pd.DataFrame({
            "Name": names,
            "Protein": proteins,
            "Shape": shapes,
            "areatotal": areatotals,
            "Obj_n": obj_ns,
            "density": densities
        })

        df = pd.concat([df, pre_df], ignore_index=True)

    new_filename = "Density_results_" + str(current_time).replace(":","_").replace("-", "_") + ".csv"
    df.to_csv(os.path.join(PATH,new_filename))
    print("Process complete.")


##########################################################################################
# ARGUMENT PARSER

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Perform density calculations on image files.')

    # Add arguments
    parser.add_argument('PATH', 
                        type=str, 
                        help='The path to the directory containing image files.')
    parser.add_argument('-tech',
                        '--PROJECT_TECHNIQUE', 
                        type=str, 
                        default='STORM', 
                        help='The project technique (default: "STORM").')
    parser.add_argument('-con',
                        '--connectivity', 
                        type=int, 
                        default=3, 
                        help='The connectivity for object identification (default: 3).')
    parser.add_argument("-th",
                        '--threshold', 
                        type=float, 
                        default=0, 
                        help='The threshold value for object identification (default: 0).')
    parser.add_argument('-xy',
                        '--x_y_ratio', 
                        type=float, 
                        default=None, 
                        help='The pixel size ratio for X and Y dimensions (default: None).')
    parser.add_argument("-z",
                        '--z_ratio', 
                        type=float, 
                        default=None, 
                        help='The pixel size ratio for the Z dimension (default: None).')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call your main function with the parsed arguments
    Densiometer(args.PATH, 
                args.PROJECT_TECHNIQUE, 
                args.connectivity, 
                args.threshold, 
                args.x_y_ratio, 
                args.z_ratio)

# Execute the main function when the script is run
if __name__ == '__main__':
    main()
