import os
from pathlib import Path, PureWindowsPath
import pandas as pd
import numpy as np
from tifffile import imread

def initialize_alignment(PATH:str, Filenames:list, Preset_file:str = None, save_template:bool = False) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:

    """
    Initialize the alignment process with image data and optional presets.

    Parameters:
        PATH (str): The path to the directory containing image files.
        Filenames (list): A list of filenames for the images to be aligned.
        Preset_file (str, optional): The name of the presets file (CSV format). Default is None.
        save_template (bool, optional): Whether to save the alignment template image. Default is False.

    Returns:
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
            A list containing the following elements:
            1. image: The first image loaded from Filenames.
            2. image2: The second image loaded from Filenames.
            3. Template: A blank image for alignment.
            4. final_img: A copy of the first image for aligned results.
            5. final_img2: A copy of the second image for aligned results.
            6. SelectedROIs: An empty DataFrame to store ROI information.
    """
    Wpath = PureWindowsPath(PATH)
    PATH = Path(Wpath)
    try:
        Presets = pd.read_csv(PATH / Preset_file)
        Presets_loaded = True
    except: 
        Presets_loaded = False

    image = imread(PATH / Filenames[0]).astype(np.float32)
    image2 = imread(PATH / Filenames[1]).astype(np.float32)

    Template = np.zeros_like(image)

    if Presets_loaded:
        for i in range(Presets.shape[0]):            
            Template[Presets.frame[i], Presets.Y[i]:Presets.Y[i] + Presets.H[i], Presets.X[i]:Presets.X[i] + Presets.W[i]] = image[Presets.frame[i], Presets.Y[i]:Presets.Y[i] + Presets.H[i], Presets.X[i]:Presets.X[i] + Presets.W[i]]

    final_img = image.copy()
    final_img2 = image2.copy()

    SelectedROIs = pd.DataFrame(columns= ["frame", "X", "Y", "W", "H", "Comments"])

    print(f"The Image has {image.shape[0]} frames.")

    return [image, image2, Template, final_img, final_img2, SelectedROIs]