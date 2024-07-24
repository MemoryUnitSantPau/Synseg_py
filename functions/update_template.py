import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def update_template(Template:np.ndarray, 
                    SelectedROIs:pd.DataFrame, 
                    image:np.ndarray, 
                    layer:int, 
                    X:int, 
                    Y:int, 
                    W:int, 
                    H:int, 
                    Comment:str)->[np.ndarray, pd.DataFrame]:
    
    """
    Update a template image with a selected ROI and store ROI information in a DataFrame.

    Parameters:
        Template (np.ndarray): The template image to be updated.
        SelectedROIs (pd.DataFrame): A DataFrame to store ROI information.
        image (np.ndarray): The image containing the ROI.
        layer (int): The layer (frame) in the template to update.
        X (int): X-coordinate of the top-left corner of the ROI.
        Y (int): Y-coordinate of the top-left corner of the ROI.
        W (int): Width of the ROI.
        H (int): Height of the ROI.
        Comment (str): Comments or notes about the ROI.

    Returns:
        [np.ndarray, pd.DataFrame]: Updated template image and DataFrame with ROI information.
    """

    # Add the ROI to the Template
    Template[layer-1, Y:Y+H, X:X+W] = image[layer-1, Y:Y+H, X:X+W]

    # Save Selected ROI information to a dataframe 
    SelectedROIs = SelectedROIs.append({"frame": layer-1, "X": X, "Y": Y, "W": W, "H": H, "Comments": Comment}, ignore_index=True)

    # Show the current layer of the template
    plt.imshow(Template[layer-1])
    plt.close()

    return [Template, SelectedROIs]