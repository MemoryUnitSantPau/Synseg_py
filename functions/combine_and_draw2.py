import numpy as np



def combine_and_draw2(image1:np.ndarray, 
                      image2:np.ndarray, 
                      point1:np.ndarray, 
                      point2:np.ndarray,
                      yellow_center:bool = False) -> np.ndarray:
    """
    Combine two images and draw a line between two specified points.

    This function takes two images and draws a line between two specified points using the provided colors.

    Parameters:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
        point1 (tuple of int): The coordinates (y, x) of the first point.
        point2 (tuple of int): The coordinates (y, x) of the second point.

    Returns:
        numpy.ndarray: The combined image with a line drawn between the two points.

    Example:
        >>> image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        >>> point1 = (10, 20)
        >>> point2 = (90, 80)
        >>> combined = combine_and_draw(image1, image2, point1, point2)
        >>> plt.imshow(combined)
        >>> plt.show()
    """

    # Create a copy of the images to avoid modifying the originals
    X, Y = image1.shape
    combined_image = np.zeros((X, Y, 4), dtype=np.uint8)
    combined_image[:, :, 0] = image1*(255/np.max(image1))
    combined_image[:, :, 1] = image2*(255/np.max(image2))
    combined_image[:, :, 2] = 0
    combined_image[:, :, 3] = 0

    # Add red pixels at the specified points
    y1, x1 = map(int, point1)
    y2, x2 = map(int, point2)
    ay1, ax1 = map(int, point1)
    ay2, ax2 = map(int, point2)

    # We'll use Bresenham's line algorithm to calculate the line pixels
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        combined_image[y1, x1, :] = [1, 1, 1, 0]# White would be 0,0,0,0, but 
                                                # it would be set black like the background
                                                # when changing cmyk to rgb so i had do make
                                                # it pseudo-white
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    # Set starting and ending piont as yellow
    if yellow_center:
        combined_image[ay1, ax1, :] = [0,0,255,0]  # Set yellow color for first point
        combined_image[ay2, ax2, :] = [0,0,255,0]  # Set yellow color for second point

    return combined_image