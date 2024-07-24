import numpy as np

def combine_and_draw(image:np.ndarray, 
                     point1:tuple, 
                     point2:tuple, 
                     color:tuple) -> np.ndarray:
    """
    Combine two images and draw a colored line between two points using Bresenham's line algorithm.

    Parameters:
        image (numpy.ndarray): The input image on which to draw the line.
        point1 (tuple of int): The starting point (y, x) of the line.
        point2 (tuple of int): The ending point (y, x) of the line.
        color (tuple of int): The color of the line in the format (R, G, B).

    Returns:
        numpy.ndarray: A new image with the line drawn between point1 and point2 in the specified color.
    """
    canvas = image.copy()

    y1, x1 = map(int, point1)
    y2, x2 = map(int, point2)

    # Bresenham's line algorithm to calculate the line pixels
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        canvas[y1, x1, :] = color
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return canvas