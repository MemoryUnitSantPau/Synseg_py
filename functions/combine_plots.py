import numpy as np
import matplotlib.pyplot as plt
from functions.cmyk_to_rgb import cmyk_to_rgb

def combine_plots(A:np.ndarray, 
                  B:np.ndarray, 
                  image2:np.ndarray, 
                  image3:np.ndarray,
                  COM_dis:float,
                  PER_dis:float, 
                  Path:str)->None:
    """
    Combine grayscale images and plot them in a 1x3 grid.

    This function takes two grayscale images (A and B) and combines them into a single RGB image,
    along with two additional images (image2 and image3), and plots them in a 1x3 grid.

    Parameters:
        A (numpy.ndarray): The first grayscale image.
        B (numpy.ndarray): The second grayscale image.
        image2 (numpy.ndarray): The third image to be displayed in the middle subplot.
        image3 (numpy.ndarray): The fourth image to be displayed in the right subplot.
        Path (str): The path where the combined plot will be saved.

    Returns:
        None

    Example:
        >>> image1 = np.random.rand(100, 100)
        >>> image2 = np.random.rand(100, 100)
        >>> image3 = np.random.rand(100, 100)
        >>> combine_plots(image1, image2, image3, "combined_plot.png")
    """

    X, Y = A.shape
    image1  = np.zeros((X, Y, 4), dtype=np.uint8)

    # Set the first grayscale image to green color channel (G)
    image1 [:, :, 0] = A*(255/np.max(A))

    # Set the second grayscale image to blue color channel (B)
    image1 [:, :, 1] = B*(255/np.max(B))

    # Create a figure with 3 subplots in a 1x3 grid
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))


    ## Manage the value displayed on the final viz
    # Currently shows nm as units. te values are transformed from um to nm  also on this step.
    if COM_dis == "OVERLAP":
        CD = 0
    else:
        CD = COM_dis * 1000
    
    if PER_dis == "OVERLAP":
        PD = 0
    else:
        PD = PER_dis * 1000

    # Display the first image in the first subplot
    axes[0].imshow(cmyk_to_rgb(image1[:,:,0], image1[:,:,1], image1[:,:,2], image1[:,:,3]))
    axes[0].set_title("Thresholded Image")
    axes[1].imshow(cmyk_to_rgb(image2[:,:,0], image2[:,:,1], image2[:,:,2], image2[:,:,3]))
    axes[1].set_title("Distance between centroids ("+str(round(CD,2))+" nm)")
    axes[2].imshow(cmyk_to_rgb(image3[:,:,0], image3[:,:,1], image3[:,:,2], image3[:,:,3]))
    axes[2].set_title("Synaptic clift measurement ("+str(round(PD,2))+" nm)")

    # Adjust layout to avoid overlap between subplots
    plt.tight_layout()
    plt.savefig(Path)
    plt.close()