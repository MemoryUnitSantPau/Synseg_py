import numpy as np
import matplotlib.pyplot as plt

def roi_check(image:np.ndarray,
              template:np.ndarray,
              layer:int,
              X:int,
              Y:int,
              H:int,
              W:int) -> None:
    """
    Display a frame, its corresponding ROI, and the template side by side.

    Parameters:
        frame (numpy.ndarray): The frame image to be displayed.
        roi (numpy.ndarray): The ROI (region of interest) image to be displayed.
        template (numpy.ndarray): The template image to be displayed.

    Returns:
        None
    """

    temp = template.copy()
    temp[layer-1, Y:Y+H, X:X+W] = 255
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    axs[0].imshow(image[layer-1], cmap="gray")
    axs[0].set_title('Frame')
    
    axs[1].imshow(image[layer-1, Y:Y+H, X:X+W], cmap="gray")
    axs[1].set_title('ROI')
    
    axs[2].imshow(temp[layer-1], cmap="gray")
    axs[2].set_title('Template')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    return