import numpy as np

def cmyk_to_rgb(c:np.ndarray, 
                m:np.ndarray, 
                y:np.ndarray, 
                k:np.ndarray, 
                cmyk_scale:int = 255, 
                rgb_scale:int = 255) -> np.ndarray:

    """
    Convert CMYK color values to RGB color values.

    This function takes CMYK color values (cyan, magenta, yellow, and black) and
    converts them to RGB color values. The CMYK values are expected to be in the range
    [0, cmyk_scale], where cmyk_scale is the maximum value for CMYK components.
    The resulting RGB values are returned as a NumPy array.

    Args:
        c (array): Cyan component of the CMYK color.
        m (array): Magenta component of the CMYK color.
        y (array): Yellow component of the CMYK color.
        k (array): Black component of the CMYK color.
        cmyk_scale (int, optional): Maximum value for CMYK components. Default is 255.
        rgb_scale (int, optional): Maximum value for RGB components. Default is 255.

    Returns:
        numpy.ndarray: An RGB array representing the converted color array.

    Example:
        c = 100
        m = 50
        y = 0
        k = 0
        rgb_color = cmyk_to_rgb(c, m, y, k)
        print(rgb_color)  # Output: [127 191 255]

    Note:
        This function assumes that the input CMYK values are within the specified range.
        The CMYK to RGB conversion may not be perfect due to color space differences.

    Reference:
        Original JavaScript implementation: http://www.javascripter.net/faq/rgb2cmyk.htm
    """

    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))

    rgb = np.array([r, g, b], dtype=np.uint8).transpose(1,2,0)  # Create the RGB array
    mask = np.all(rgb == [255, 255, 255], axis=-1)  # Identify white pixels
    rgb[mask] = [0, 0, 0]

    return rgb