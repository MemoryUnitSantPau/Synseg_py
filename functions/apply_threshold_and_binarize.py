import numpy as np

def apply_threshold_and_binarize(image_frame:np.ndarray, th_percentage:float):
    """
    Applies thresholding and binarization to an input image frame.

    Parameters:
    image_frame (numpy.ndarray): Input image frame as a NumPy array.
    th_precentage (float): Input percentage of thresholding.

    Returns:
    numpy.ndarray: Binarized image array where values above a threshold are set to 255 and others to 0.
    """
    threshold = np.max(image_frame) * th_percentage
    image_bin = (image_frame > threshold) * 255
    return image_bin
