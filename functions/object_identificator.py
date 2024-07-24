import numpy as np
import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage.measure import regionprops_table

def object_identificator(closing:np.ndarray, 
                         frame:int, 
                         min_area:int)->tuple:
    """
    Identify and analyze objects in a preprocessed image frame.

    Parameters:
        closing (numpy.ndarray): The preprocessed image frame.
        frame (int): The frame index associated with the image.
        min_area (int): The minimum area for an object to be considered.

    Returns:
        Tuple[numpy.ndarray, pandas.DataFrame]: A tuple containing the labeled image of identified objects
        and a DataFrame with object properties (centroid, area, bounding box, label, and frame).
    """
    image_labeled = label(closing)[0]

    # Calculate region proprieties
    image_proprieties = regionprops_table(image_labeled, properties = ( 'centroid',
                                                                        'area',
                                                                        'bbox',
                                                                        'label'))
    image_proprieties = pd.DataFrame(image_proprieties)
    image_proprieties['Frame'] = np.repeat(frame, len(image_proprieties), axis=0)
    image_proprieties = image_proprieties[image_proprieties['area'] > min_area]

    return image_labeled, image_proprieties