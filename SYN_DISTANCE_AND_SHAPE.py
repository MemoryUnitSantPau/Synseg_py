import argparse
import os
import re
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from pathlib import Path, PureWindowsPath
from math import sqrt
import pandas as pd
from skimage import measure
import warnings
import diplib as dip
import matplotlib.pyplot as plt
from functions.extract_unique_texts_by_pattern import extract_unique_texts_by_pattern
from functions.group_files_by_common_part3 import group_files_by_common_part
from functions.detect_shapes import detect_shapes
from functions.detect_smooth_shapes import detect_smooth_shapes
from functions.calculate_shape_similarity import calculate_shape_similarity


warnings.simplefilter(action='ignore', category=FutureWarning)

#######################################################################

def distance_and_shape(PATH:str, um_pix_ratio:float = 0.4274 , plot_save = False):
    """
    Processes images in the specified directory, computes distances and shapes,
    and saves the results to CSV files.

    Args:
        PATH (str): The directory containing the image files to be processed.
        um_pix_ratio (float, optional): The ratio to convert pixels to micrometers. Default is 0.4274.
        plot_save (bool, optional): Flag to indicate whether to save plots. Default is False.

    Returns:
        None

    Notes:
        - The directory specified by PATH should contain images with a common name and channel name in brackets.
          Example: Common_name(SYPH).tif, Common_name(PSD95).tif. Only two channels are expected per common name.
        - The function processes the images, extracts shapes and computes distances between centroids of corresponding shapes
          in different channels.
        - The results are saved to two CSV files:
            1. 'results_per_frame.csv': Contains the computed properties for each frame.
            2. 'results_per_protein.csv': Contains the computed properties grouped by protein.

    Example:
        >>> distance_and_shape("/path/to/images", 0.5, True)
    """
    # Convert the given path to a format compatible with the operating system
    Wpath = PureWindowsPath(PATH)
    PATH = Path(Wpath)

    # Uncoment for linux usage
    #PATH = Path(Wpath)

    # Initialize counters and lists
    Frame_counter = 0
    Good_Frames = 0
    file_list = [file for file in os.listdir(PATH) if file.endswith(".tif")]

    # Extract unique text patterns between parentheses from filenames
    unique_texts_between_parentheses = extract_unique_texts_by_pattern(file_list)

    # Group files based on the common part of their names
    Files = group_files_by_common_part(file_list, unique_texts_between_parentheses)
    missing_matches = list()
    df = pd.DataFrame()

    # Process each group of files
    for Filegroup in Files:

        Filegroup = sorted(Filegroup)
        if len(Filegroup) == 1:
            # If there is only one file in the group, add it to missing matches
            missing_matches = missing_matches.append(Filegroup[0])
            continue

        # Read the images for each channel
        I = tifffile.imread(os.path.join(PATH, Filegroup[0]))
        I2 = tifffile.imread(os.path.join(PATH, Filegroup[1]))
        CH1_name = str(extract_unique_texts_by_pattern([Filegroup[0]])[0])
        CH2_name = str(extract_unique_texts_by_pattern([Filegroup[1]])[0])

        Dist = None
        frame_number = I.shape[0] if len(I.shape) == 3 else 1
        for frame in range(frame_number):
            Frame_counter += 1
            
            miss_A = False
            miss_B = False
            
            # Threshold and label the images for the current frame
            if frame_number == 1:
                threshold_A = np.max(I)*0.2
                threshold_B = np.max(I2)*0.2
                A = label(I>threshold_A)[0]
                B = label(I2>threshold_B)[0]

            else:
                threshold_A = np.max(I[frame])*0.2
                threshold_B = np.max(I2[frame])*0.2
                A = label(I[frame]>threshold_A)[0]
                B = label(I2[frame]>threshold_B)[0]

            # Measure region properties
            PropA = measure.regionprops(A)
            PropB = measure.regionprops(B)
            
            # Find the largest object in each image
            try:
                Index_A = max(PropA, key=lambda region: region.area).label
            except:
                Index_A = 1
                miss_A = True
            try:
                Index_B = max(PropB, key=lambda region: region.area).label
            except:
                Index_B = 1
                miss_B = True

            # Compute properties if both objects are found
            if miss_A == False and miss_B == False:

                Good_Frames += 1
                
                # Extract the center of mass of each object
                COM_A = center_of_mass(A == Index_A)
                COM_B = center_of_mass(B == Index_B)

                # Calculate the distance in the x and y components
                X = abs(COM_A[0] - COM_B[0]) if COM_A[0] > COM_B[0] else abs(COM_B[0] - COM_A[0])
                Y = abs(COM_A[1] - COM_B[1]) if COM_A[1] > COM_B[1] else abs(COM_B[1] - COM_A[1])
                
                # Obtain the distance between centers
                COM_Dist = sqrt(X**2 + Y**2)
                COM_Dist_um = COM_Dist * um_pix_ratio

                POSB = np.where(B == Index_B)
                POSA = np.where(A == Index_A)
                
                # Extract shape features and similarity
                A_shape = detect_shapes(A == Index_A)
                B_shape = detect_shapes(B == Index_B)
                A_smooth = detect_smooth_shapes(A == Index_A)
                B_smooth = detect_smooth_shapes(B == Index_B)
                A_similarity = calculate_shape_similarity(A == Index_A)
                B_similarity = calculate_shape_similarity(B == Index_B)

                # Measure object shape properties using diplib
                labels_A = dip.Label(A == Index_A)
                msr_A = dip.MeasurementTool.Measure(labels_A, features=["Perimeter", "Size", "Roundness", "Circularity"])
                labels_B = dip.Label(B == Index_B)
                msr_B = dip.MeasurementTool.Measure(labels_B, features=["Perimeter", "Size", "Roundness", "Circularity"])

                # Create data dictionary for the current frame
                data = {
                    'Name': Filegroup[0].split("(")[0],
                    'Frame': frame,
                    'Centroid_' + CH1_name: COM_A,
                    'Centroid_' + CH2_name:COM_B,
                    'Center_Distance_px': COM_Dist,
                    'Center_Distance_um': COM_Dist_um,
                    'Shape1_' + CH1_name : A_shape,
                    'Shape1_' + CH2_name : B_shape,
                    'Perimeter_' + CH1_name : msr_A["Perimeter"][1],
                    'Perimeter_' + CH2_name : msr_B["Perimeter"][1],
                    'Size_' + CH1_name : msr_A["Size"][1],
                    'Size_' + CH2_name : msr_B["Size"][1],
                    'Roundness_' + CH1_name : msr_A["Roundness"][1],
                    'Roundness_' + CH2_name : msr_B["Roundness"][1],
                    'Circularity_' + CH1_name : msr_A["Circularity"][1],
                    'Circularity_' + CH2_name : msr_B["Circularity"][1]
                    }   
                
                # Extract additional properties for the objects
                Prop_only_A = measure.regionprops(label(A == Index_A)[0])
                for prop in Prop_only_A[0]:
                        if prop not in ['image', 
                                "coords", 
                                'image_filled', 
                                'image_convex', 
                                'euler_number', 
                                'extent', 
                                'filled_area', 
                                'filled_image', 
                                'inertia_tensor', 
                                'inertia_tensor_eigvals', 
                                'slice',
                                'moments',
                                'moments_central',
                                'moments_hu',
                                'moments_normalized']:
                            data[prop + "_" + CH1_name] = Prop_only_A[0][prop]

                Prop_only_B = measure.regionprops(label(B == Index_B)[0])

                for prop in Prop_only_B[0]:
                        if prop not in ['image', 
                                "coords", 
                                'image_filled', 
                                'image_convex', 
                                'euler_number', 
                                'extent', 
                                'filled_area', 
                                'filled_image', 
                                'inertia_tensor', 
                                'inertia_tensor_eigvals', 
                                'slice',
                                'moments',
                                'moments_central',
                                'moments_hu',
                                'moments_normalized']:
                            data[prop + "_" + CH2_name] = Prop_only_B[0][prop]

            # Handle cases where one of the objects is missing
            elif miss_A == True and miss_B == False:

                POSB = np.where(B == Index_B)

                B_shape = detect_shapes(B == Index_B)
                B_smooth = detect_smooth_shapes(B == Index_B)
                B_similarity = calculate_shape_similarity(B == Index_B)

                labels = dip.Label(B == Index_B)
                msr = dip.MeasurementTool.Measure(labels, features=["Perimeter", "Size", "Roundness", "Circularity"])

                data = {
                    'Name': Filegroup[0].split("(")[0],
                    'Frame': frame,
                    'Centroid_'+CH1_name: "null",
                    'Centroid_'+CH2_name: COM_B,
                    'Center_Distance_px': "null",
                    'Center_Distance_um': "null",
                    'Shape1_' + CH1_name : "null",
                    'Shape1_' + CH2_name : B_shape,
                    'Perimeter_' + CH1_name : "null",
                    'Perimeter_' + CH2_name : msr["Perimeter"][1],
                    'Size_' + CH1_name : "null",
                    'Size_' + CH2_name : msr["Size"][1],
                    'Roundness_' + CH1_name : "null",
                    'Roundness_' + CH2_name : msr["Roundness"][1],
                    'Circularity_' + CH1_name : "null",
                    'Circularity_' + CH2_name : msr["Circularity"][1]
                    }
                
                
                Prop_only_B = measure.regionprops(label(B == Index_B)[0])

                for prop in Prop_only_B[0]:
                    if prop not in ['image', 
                            "coords", 
                            'image_filled', 
                            'image_convex', 
                            'euler_number', 
                            'extent', 
                            'filled_area', 
                            'filled_image', 
                            'inertia_tensor', 
                            'inertia_tensor_eigvals', 
                            'slice',
                            'moments',
                            'moments_central',
                            'moments_hu',
                            'moments_normalized']:
                        data[prop + "_" + CH1_name] = "null"

                for prop in Prop_only_B[0]:
                    if prop not in ['image', 
                            "coords", 
                            'image_filled', 
                            'image_convex', 
                            'euler_number', 
                            'extent', 
                            'filled_area', 
                            'filled_image', 
                            'inertia_tensor', 
                            'inertia_tensor_eigvals', 
                            'slice',
                            'moments',
                            'moments_central',
                            'moments_hu',
                            'moments_normalized']:
                        data[prop + "_" + CH2_name] = Prop_only_B[0][prop]



            elif miss_A == False and miss_B == True:

                POSA = np.where(A == Index_A)

                A_shape = detect_shapes(A == Index_A)
                A_smooth = detect_smooth_shapes(A == Index_A)
                A_similarity = calculate_shape_similarity(A == Index_A)

                labels = dip.Label(A == Index_A)
                msr = dip.MeasurementTool.Measure(labels, features=["Perimeter", "Size", "Roundness", "Circularity"])
                
                data = {
                    'Name': Filegroup[0].split("(")[0],
                    'Frame': frame,
                    'Centroid_'+CH1_name: COM_A,
                    'Centroid_'+CH2_name:"null",
                    'Center_Distance_px': "null",
                    'Center_Distance_um': "null",
                    'Shape1_' + CH1_name : A_shape,
                    'Shape1_' + CH2_name : "null",
                    'Perimeter_' + CH1_name : msr["Perimeter"][1],
                    'Perimeter_' + CH2_name : "null",
                    'Size_' + CH1_name : msr["Size"][1],
                    'Size_' + CH2_name : "null",
                    'Roundness_' + CH1_name : msr["Roundness"][1],
                    'Roundness_' + CH2_name : "null",
                    'Circularity_' + CH1_name : msr["Circularity"][1],
                    'Circularity_' + CH2_name : "null"
                    }
                
                Prop_only_A = measure.regionprops(label(A == Index_A)[0])

                for prop in Prop_only_A[0]:
                    if prop not in ['image', 
                            "coords", 
                            'image_filled', 
                            'image_convex', 
                            'euler_number', 
                            'extent', 
                            'filled_area', 
                            'filled_image', 
                            'inertia_tensor', 
                            'inertia_tensor_eigvals', 
                            'slice',
                            'moments',
                            'moments_central',
                            'moments_hu',
                            'moments_normalized']:
                        data[prop + "_" + CH1_name] = Prop_only_A[0][prop]

                for prop in Prop_only_A[0]:
                    if prop not in ['image', 
                            "coords", 
                            'image_filled', 
                            'image_convex', 
                            'euler_number', 
                            'extent', 
                            'filled_area', 
                            'filled_image', 
                            'inertia_tensor', 
                            'inertia_tensor_eigvals', 
                            'slice',
                            'moments',
                            'moments_central',
                            'moments_hu',
                            'moments_normalized']:
                        data[prop + "_" + CH2_name] = "null"

            else :
                
                bypas_img = np.zeros((3,3))
                bypas_img[1,1] = 1
                data = {
                    'Name': Filegroup[0].split("(")[0],
                    'Frame': frame,
                    'Centroid_'+CH1_name: "null",
                    'Centroid_'+CH2_name:"null",
                    'Center_Distance_px': "null",
                    'Center_Distance_um': "null",
                    'Shape1_' + CH1_name : "null",
                    'Shape1_' + CH2_name : "null",
                    'Perimeter_' + CH1_name : "null",
                    'Perimeter_' + CH2_name : "null",
                    'Size_' + CH1_name : "null",
                    'Size_' + CH2_name : "null",
                    'Roundness_' + CH1_name : "null",
                    'Roundness_' + CH2_name : "null",
                    'Circularity_' + CH1_name : "null",
                    'Circularity_' + CH2_name : "null"
                    }

                Prop_only_A = measure.regionprops(label(bypas_img)[0])

                for prop in Prop_only_A[0]:
                    if prop not in ['image', 
                            "coords", 
                            'image_filled', 
                            'image_convex', 
                            'euler_number', 
                            'extent', 
                            'filled_area', 
                            'filled_image', 
                            'inertia_tensor', 
                            'inertia_tensor_eigvals', 
                            'slice',
                            'moments',
                            'moments_central',
                            'moments_hu',
                            'moments_normalized']:
                        data[prop + "_" + CH1_name] = "null"

                Prop_only_B = measure.regionprops(label(bypas_img)[0])

                for prop in Prop_only_B[0]:
                    if prop not in ['image', 
                            "coords", 
                            'image_filled', 
                            'image_convex', 
                            'euler_number', 
                            'extent', 
                            'filled_area', 
                            'filled_image', 
                            'inertia_tensor', 
                            'inertia_tensor_eigvals', 
                            'slice',
                            'moments',
                            'moments_central',
                            'moments_hu',
                            'moments_normalized']:
                        data[prop + "_" + CH2_name] = "null"

            
            df2 = pd.DataFrame([data])
            df = pd.concat([df, df2], ignore_index=True)        

    df['Circularity_' + CH1_name] = df['Circularity_' + CH1_name].apply(lambda x: x[0])
    df['Roundness_' + CH1_name] = df['Roundness_' + CH1_name].apply(lambda x: x[0])
    df['Size_' + CH1_name] = df['Size_' + CH1_name].apply(lambda x: x[0])
    df['Perimeter_' + CH1_name] = df['Perimeter_' + CH1_name].apply(lambda x: x[0])
    df['Circularity_' + CH2_name] = df['Circularity_' + CH2_name].apply(lambda x: x[0])
    df['Roundness_' + CH2_name] = df['Roundness_' + CH2_name].apply(lambda x: x[0])
    df['Size_' + CH2_name] = df['Size_' + CH2_name].apply(lambda x: x[0])
    df['Perimeter_' + CH2_name] = df['Perimeter_' + CH2_name].apply(lambda x: x[0])

    df.to_csv(os.path.join(PATH,"results_per_frame.csv"))

    subset_not_syph = df.filter(regex='^(?!.*' + CH1_name + ').*$') 
    subset_not_syph.columns = subset_not_syph.columns.str.replace('_' + CH2_name, '')
    subset_not_syph['Protein'] = CH2_name

    subset_not_psd = df.filter(regex='^(?!.*' + CH2_name + ').*$')
    subset_not_psd.columns = subset_not_psd.columns.str.replace('_' + CH1_name, '')
    subset_not_psd['Protein'] = CH1_name

    rejoined = pd.concat([subset_not_syph, subset_not_psd], ignore_index=True)
    rejoined = rejoined.sort_values(by=['Name', 'Protein'])
    rejoined.to_csv(os.path.join(PATH,"results_per_protein.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process image pairs and calculate distances and shape properties.')
    parser.add_argument('path', type=str, help='The path to the directory containing .tif image files.')
    parser.add_argument('--um_pix_ratio', type=float, default=0.4274, help='The pixel to micrometer conversion ratio. Default is 0.4274.')
    parser.add_argument('--plot_save', action='store_true', help='Save plots of the results. Default is False.')

    args = parser.parse_args()

    distance_and_shape(args.path, args.um_pix_ratio, args.plot_save)