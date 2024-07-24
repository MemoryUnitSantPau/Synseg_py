import os
import numpy as np
import pandas as pd
from functions.apply_threshold_and_binarize import apply_threshold_and_binarize
from functions.cmyk_to_rgb import cmyk_to_rgb
from tifffile import imwrite
from pathlib import Path, PureWindowsPath

def ROI_MAP(image:np.ndarray, 
            image2:np.ndarray, 
            df:pd.DataFrame, 
            mask:np.ndarray, 
            filename:str, 
            PATH:str, 
            th_percentage:float)->None:
    """
    Generate a map of regions of interest (ROIs) in image frames.

    Parameters:
        image (numpy.ndarray): The first input image data (frame_n, x, y).
        image2 (numpy.ndarray): The second input image data (frame_n, x, y).
        df (pandas.DataFrame): DataFrame containing ROI data (big_roi, Frame).
        mask (numpy.ndarray): The mask data (frame_n, x, y) for visual analysis.
        filename (str): The name of the output file.
        PATH (str): The path to the directory where the output file will be saved.
        th_percentage (float): The threshold percentage for binarization.

    Returns:
        None
    """
    Wpath = PureWindowsPath(PATH)
    PATH = Path(Wpath)
    mask_projection = np.sum(mask, axis = 0)

    # ROI SIZE
    frame_n, x, y = image.shape
    MAP = np.zeros((frame_n, x, y, 3))
    Viz = np.zeros((frame_n, x, y, 4))

    for frame in range(frame_n):
        
        Viz[frame, :, :, 0] = apply_threshold_and_binarize(image[frame, :, :], th_percentage)
        Viz[frame, :, :, 1] = apply_threshold_and_binarize(image2[frame, :, :], th_percentage)

        for roi, frame2 in zip(df["big_roi"], df["Frame"]):
            
            i_x = int(roi[0])
            f_x = int(roi[1])
            i_y = int(roi[2])
            f_y = int(roi[3])
 
            if f_x - i_x > 1000 or f_y - i_y > 1000:
                continue

            if mask_projection[i_x:f_x, i_y:f_y].size > 0:
                if np.mean(mask_projection[i_x:f_x, i_y:f_y]) > 1.5:
                    square_color = [0, 0, 255, 0]
                else:
                    square_color = [0, 127, 255, 0]
            else:
                square_color = [255, 0, 0, 0]
            
            # DRAW BOX 
            Viz[int(frame2), i_x:f_x, i_y, :] = square_color
            Viz[int(frame2), i_x:f_x, f_y, :] = square_color
            Viz[int(frame2), i_x, i_y:f_y, :] = square_color
            Viz[int(frame2), f_x, i_y:f_y, :] = square_color

        Cyan = Viz[frame, :, :, 0] 
        Magenta = Viz[frame, :, :, 1]
        Yellow = Viz[frame, :, :, 2]
        Black = Viz[frame, :, :, 3]
        PreMAP = cmyk_to_rgb(Cyan, Magenta, Yellow, Black)
        MAP[frame,:,:,:] = PreMAP

    viz_filename = str(filename)+"ROI_MAP.tif"
    imwrite(PATH / "ROI_MAPS" / viz_filename, MAP.astype(np.uint8))
