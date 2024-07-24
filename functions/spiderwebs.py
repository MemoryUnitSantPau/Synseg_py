import numpy as np
import pandas as pd
from tifffile import imwrite
from pathlib import Path, PureWindowsPath
import os

from functions.apply_threshold_and_binarize import apply_threshold_and_binarize 
from functions.combine_and_draw import combine_and_draw
from functions.cmyk_to_rgb import cmyk_to_rgb 

def spiderwebs(image:np.ndarray, 
               image2:np.ndarray, 
               df:pd.DataFrame, 
               mask:np.ndarray, 
               filename:str, 
               PATH:str, 
               th_percentage:float) -> None:
    """
    Generate spiderweb-like visualizations from image data.

    Parameters:
        image (numpy.ndarray): The first input image data (frame_n, x, y).
        image2 (numpy.ndarray): The second input image data (frame_n, x, y).
        df (pandas.DataFrame): DataFrame containing object data.
        mask (numpy.ndarray): The mask data (frame_n, x, y) for visual analysis.
        filename (str): The name of the output file.
        PATH (str): The path to the directory where the output file will be saved.
        th_percentage (float): The threshold percentage for binarization.

    Returns:
        None
    """
    print(df.columns)

    Wpath = PureWindowsPath(PATH)
    PATH = Path(Wpath)

    frame_n, x, y = image.shape
    mask_projection = np.sum(mask, axis = 0)
    Viz = np.zeros((frame_n, x, y, 4))
    MAP = np.zeros((frame_n, x, y, 3))

    for frame in range(frame_n):
        
        Viz[frame, :, :, 0] = apply_threshold_and_binarize(image[frame, :, :], th_percentage)
        Viz[frame, :, :, 1] = apply_threshold_and_binarize(image2[frame, :, :], th_percentage)

    for obj_i in range(len(df)):

        data = df.loc[obj_i]
        frame = int(data['Frame'])

        roi = data['big_roi']
        i_x = int(roi[0])
        f_x = int(roi[1])
        i_y = int(roi[2])
        f_y = int(roi[3])

        if mask_projection[i_x:f_x, i_y:f_y].size > 0:
            if np.mean(mask_projection[i_x:f_x, i_y:f_y]) > 1.5:
                square_color = [0, 0, 255, 0]
            else:
                square_color = [0, 127, 255, 0]
        else:
            square_color = [255, 0, 0, 0]
         
        Viz[frame, :,:,:] = combine_and_draw(Viz[frame, :,:,:], data['com'], data['com2'], square_color)

    for frame in range(frame_n):

        Cyan = Viz[frame, :, :, 0] 
        Magenta = Viz[frame, :, :, 1]
        Yellow = Viz[frame, :, :, 2]
        Black = Viz[frame, :, :, 3]
        PreMAP = cmyk_to_rgb(Cyan, Magenta, Yellow, Black)
        MAP[frame,:,:,:] = PreMAP

    viz_file = str(filename)+"ROI_MAP_SPIDERWEB.tif"
    
    imwrite(PATH / "ROI_MAPS" / viz_file, MAP.astype(np.uint8))
