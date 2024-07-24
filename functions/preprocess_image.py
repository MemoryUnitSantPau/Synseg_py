import numpy as np
import cv2

def preprocess_image(image:np.ndarray, 
                     frame:int, 
                     th_percentage:float, 
                     kernel_3x3:np.ndarray) -> np.ndarray:
    """
    Preprocesses a single frame of an image for further analysis.

    Parameters:
        image (numpy.ndarray): The input image data.
        frame (int): The frame index to process.
        th_percentage (float): The threshold percentage for binarization.
        kernel_3x3 (numpy.ndarray): The 3x3 kernel for morphological operations.

    Returns:
        numpy.ndarray: The preprocessed image frame after thresholding, opening, and closing operations.
    """
    image_frame = image[frame]
    image_frame = image_frame*(255/np.max(image_frame))
    threshold = np.max(image_frame)*th_percentage
    image_bin = (image_frame > threshold)*255
    opening = cv2.morphologyEx(image_bin.astype(np.uint8), cv2.MORPH_OPEN, kernel_3x3)
    closing = cv2.morphologyEx(opening.astype(np.uint8), cv2.MORPH_CLOSE, kernel_3x3)
    return closing * (255 / np.max(closing))

