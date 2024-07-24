import os
import ants
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tifffile import imwrite

def save_results(PATH:str,
                 Filenames:list, 
                 SelectedROIs:pd.DataFrame, 
                 Template:np.ndarray, 
                 image:np.ndarray, 
                 image2:np.ndarray, 
                 final_img:np.ndarray, 
                 final_img2:np.ndarray, 
                 save_template:bool= False, 
                 name:str = "SelectedROIs")-> None:

    """
    Save various results including CSV files, aligned images, and visualizations.

    Parameters:
        PATH (str): The path to the directory for saving results.
        Filenames (list): A list of filenames.
        SelectedROIs (pd.DataFrame): A DataFrame containing ROI information.
        Template (np.ndarray): The template image.
        image (np.ndarray): The first image.
        image2 (np.ndarray): The second image.
        final_img (np.ndarray): The final aligned image for the first image.
        final_img2 (np.ndarray): The final aligned image for the second image.
        save_template (bool, optional): Whether to save the template image (default: False).
        name (str, optional): The name for the CSV file (default: "SelectedROIs").

    Returns:
        None
    """

    if not os.path.exists(os.path.join(PATH, name +".csv")):
        SelectedROIs.to_csv(os.path.join(PATH, name +".csv"), index= False)
    else:
        counter = 0
        newname = name + "(" + str(counter) + ")"
        while os.path.exists(os.path.join(PATH, newname + ".csv")):
            counter = counter + 1
        SelectedROIs.to_csv(os.path.join(PATH, newname +".csv"), index= False)



    # Set the middle frame from the Template as Fixed image for the registration
    fixed = ants.from_numpy(Template[math.ceil(image.shape[0]/2)])

    # Iterate Over the frames to calculate the alignment 
    frame_num = 1 if len(image.shape) == 2 else image.shape[0]
    for i in range(0,frame_num):
        moving = ants.from_numpy(Template[i,:,:])

        # Calculate the matrix of transformation
        mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='Rigid' )
        warped_moving = mytx['warpedmovout']

        # Apply the calculated transformations to all the corrsponding frames of the original images.
        final_img[i] = ants.apply_transforms(fixed=ants.from_numpy(image[math.ceil(image.shape[0]/2)]), moving=ants.from_numpy(image[i]),transformlist=mytx['fwdtransforms']).numpy()
        final_img2[i] = ants.apply_transforms(fixed=ants.from_numpy(image2[math.ceil(image.shape[0]/2)]), moving=ants.from_numpy(image2[i]),transformlist=mytx['fwdtransforms']).numpy()

    # Transpose the image dimensions for plotting
    final_img_transposed = final_img.transpose(1, 2, 0)

    # Create a figure with a single subplot
    fig, ax = plt.subplots()

    # Plot the RGB image
    ax.imshow(final_img_transposed, aspect='auto')
    ax.set_title('Color Channels Overlap')

    # Display the figure
    plt.show()
    plt.close()

    if save_template:
        imwrite(os.path.join( PATH, Filenames[0].replace(".tif", "_template.tif")), Template)

    imwrite(os.path.join(PATH, Filenames[0].replace(".tif", "_aligned.tif")),final_img.astype(np.uint16), photometric= "minisblack")
    imwrite(os.path.join(PATH, Filenames[1].replace(".tif", "_aligned.tif")),final_img2.astype(np.uint16), photometric= "minisblack")
