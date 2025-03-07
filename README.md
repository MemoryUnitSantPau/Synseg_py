# SYNAPTIC MEASUREMENTS

## Introduction

This repository contains the scripts used to perform segmentation, density, and distance analysis of proteins in STORM images.

The scripts were used in __Enhancing lateral resolution using direct stochastic optical reconstruction microscopy (dSTORM) to unravel synaptic tau pathology in Alzheimer’s disease__, which can be found at [10.1111/nan.70010](https://onlinelibrary.wiley.com/doi/full/10.1111/nan.70010).

## Installation

To run the following scripts it is needed to use python and to have the required packages installed.

It is recomended to use python through anaconda or miniconda.

To install the required packages use the following command from the downloaded repository folder.

`pip install -r requirements.txt`

## Scripts

### SYN_SEGMENTATOR

The image segmentation process begins by creating a new directory to store the segmented data and identifying all TIFF image files within the specified source directory. The script iterates through each image, loading it and checking if it's a binary image (only containing 0s and 1s). If so, it skips processing that image as it likely isn't a microscopy image.

For valid microscopy images, several preprocessing steps are applied. First, the intensity values are converted to floating-point numbers between 0 and 1. Then, a local mean filter is employed to calculate the average intensity within a user-defined window around each pixel. This local mean is subtracted from the original value, highlighting object boundaries.

A threshold is then calculated based on the local mean and a factor provided by the user. This threshold is used to create a binary image by comparing each pixel's intensity in the preprocessed image. Pixels exceeding the threshold are assigned 1 (foreground/object), while those below are assigned 0 (background). The binary image is inverted, so objects appear white on a black background.

An empty 3D array is created to store the segmented objects. The script iterates through each image slice (representing different depths) and uses a function to identify and label connected objects in the binary image. This assigns a unique label to each group of foreground pixels, separating objects.

The script then iterates through each labeled object and compares its size (total number of pixels) to a minimum size threshold. If an object meets this criterion, the coordinates of all its constituent voxels (3D pixels) are added to the 3D array, accumulating the identified objects across all slices and building a 3D representation.

A new 3D image is created with the same dimensions as the original but with intensity values corresponding to object membership. The script iterates through each voxel in this new image. If the corresponding voxel's coordinates are present in the 3D array (indicating it belongs to a segmented object), the original image intensity value from that location is copied. Otherwise, the intensity value in the new image is set to 0. This effectively segments the objects based on the selection criteria.

Two separate TIFF images are then saved: the segmented image containing the isolated objects and a mask image where objects are represented by white pixels on a black background.

Finally, the script creates a text file storing the values of all function arguments used for future reference.


#### INPUTS

- path : Path to the directory containing input images
- ws : Window size for local mean or median filter
- r : Factor for calculating the threshold in the filter
- method : Method for filtering
- min_surf : Minimum 2D surface of an object
- min_size : Minimum 3D volume of an object
- max_size : Maximum 3D volume of an object
- th1 : Minimum size of an object
- th2 : Minimum size of an object

#### OUTPUTS

- Identified objects:

    - 'XXXX_join2D.tif' : Image segmented per frame 

    - 'XXXX_join3D_segmented.tif' Segmented image stack filtered to keep only objects present in consecutive frames.

    - 'XXXX_join3D_mask.tif' Merge of the original image and the join3D segmented image to obtain the segmented image with the original intensity values on the signal pixels.

- Filtered objects:

    - 'XXXX_join2D2.tif'

    - 'XXXX_join3D2_segmented.tif'

    - 'XXXX_join3D2_mask.tif'

#### Example

`python SEGMENTATION_FOR_DENSITY.py /path/to/files -tech STORM -nc SYPH -c SYPH PSD95`

### SYN_DENSITY

The method described herein details the process of performing density calculations on image files utilizing Python programming language libraries. The primary functionalities involve image processing, object identification and data analysis.

The function accepts several parameters, including the directory path containing image files, project technique, neuropil channel identifier, connectivity for object identification, intensity threshold for object identification, and pixel size ratios. This function iterates through each image file in the specified directory, performing the following procedure.

The function begins by checking whether the user has provided custom values for the pixel size ratios. If not, default values are assigned based on the specified project technique. For instance, if the project technique is "STORM," the pixel size ratios are set to 0.008 for the X and Y dimensions. The z dimention will depend on the type of sample in use (for Array Tomography samples z = 0.07).

The code obtains a list of the image files within the specified directory, and these filenames are stored in a list. A loop iterates through each image file in the list, performing the following operations.

The image is read using the `tifffile` library and is then converted to a float32 format, ensuring consistency in data type for subsequent processing steps. The image pixel volume is calculated and stored.

For each frame of the image, utilizing the morphological operations from the OpenCV library (`cv2`), the algorithm performs a closing filtering to remove noise using a 3x3 pixel kernel. The frame is then segmented using the ratio of the maximum intensity established by the user, and using SciPy, it identifies the objects within the frame. The total objects detected are calculated by adding the identified objects in each frame using NumPy.

Depending on the requirements, the complete segmented image stack is processed as a whole using SciPy and a kernel determined by the connectivity established by the user, detecting the objects in contact on the same layer or within the volume of the stack.

Once all the images are processed, the density for each of them is calculated using the objects identified in the whole image with the specified connectivity and the total pixel volume, and the result is transformed to objects per µm³ using the given ratios for the X, Y, and Z dimensions.

The results are finally stored in a CSV file for further processing and analysis.

#### INPUTS

- PATH : The path to the directory containing image files.
    - The program expects the filenames to have the channel indicated between the first and second underscores. (001_PSD95_segmentation.tif, 001_SYPH_segmentation.tif)
- tech (PROJECT_TECHNIQUE) : The project imaging technique used to set predefined parameters of size ratio.
- nc (NEUROPIL_CHANNEL) : The neuropil channel identifier.
- con (CONNECTIVITY) : The dimensions of connectivity for object identification.
- th (THRESHOLD) : The threshold percentage of the maximum intensiti value for object segmentation.
- xy (X_Y_RATIO) : The pixel size ratio for X and Y dimensions 
- z (Z_RATIO): The pixel size ratio for the Z dimension 

#### OUTPUTS

- Results_density_(date).tsv

    This file contains the results calculated on the script.
    Each row represents an image.

    The columns contain:
    - frame_(X)_ObjN : Number of objects identifyed on the frame X.
    - density_frame_(X) : Density of objects on a frame per surface of the frame.
    - areatotal : Total volume of pixels on the image.
    - Obj_n : Total number of identifyed objects (accounting for connectivity on the 3 dimensions).
    - density : density of objects per um in the total image pixel volume.

#### Example
`python SYN_DENSITY.py /path/to/files -tech STORM -con 3 -th 0`

### SYN_DISTANCE_AND_SHAPE

The distance_and_shape function performs image analysis and shape comparison on pairs of images with a common name but different channels. The function processes .tif files from a specified directory, calculates the distance between centroids of objects in different channels, and extracts various shape properties. It then saves the results in CSV files for further analysis.

The function begins by listing all .tif files in the specified directory, grouping the files by their common name to identify pairs of images for different channels. Each pair of images is read using the tifffile library and converted to a consistent format for further processing. Objects in each frame of the images are detected using a thresholding method, and the detected objects are labeled for further analysis.

Various shape properties such as perimeter, size, roundness, and circularity, eccentricity and lengths of major and minor axis are measured for each detected object using diplib and scikit image. The centroids of the detected objects are then calculated using the scipy.ndimage library, and the Euclidean distance between the centroids of objects in different channels is calculated.

The results are compiled into data frames and saved as CSV files. The function generates three output files: results_per_frame.csv, containing the results calculated per frame of each image; results_per_protein.csv, containing the results calculated per protein channel; and results_per_protein_with_clustering.csv, containing the results with clustering information.

#### INPUTS

The function accepts the following parameters:

   - PATH (str): The path to the directory containing .tif image files.
   - um_pix_ratio (float, optional): The pixel to micrometer conversion ratio. The default value is 0.4274.
   - plot_save (bool, optional): A boolean flag indicating whether to save plots of the results. The default value is False.

The function expects the filenames to have a common name with the channel name indicated in parentheses, such as Common_name(SYPH).tif and Common_name(PSD95).tif.

#### OUTPUTS
The function produces three CSV files:

   - results_per_frame.csv: This file contains the results calculated per frame of each image.
   - results_per_protein.csv: This file contains the results calculated per protein channel.

#### EXAMPLE

`python SYN_DISTANCE_AND_SHAPE.py /path/to/files --um_pix_ratio 0.4274 --plot_save`

## Requirements
Python version : Python 3.10.10

Packages and versions used in requirements.txt

## WARNINGS

The scripts are prepared to accept windows paths and might rise errors due to paths if used from a linux computer. To solve it comment the following lines from the scripts. 
    
`Wpath = PureWindowsPath(PATH)`

`PATH = Path(Wpath)`  

The image files must be .tif format and have a filename with the following pattern:
- `ImageID_ProteinChannel_Metadata.tif`
This pattern will permit the functions to recognise correctly the information obtained from the filename.


## REFERENCES

1. McKinney, Ryan. "Data Structures for Statistical Computing in Python." Proceedings of the 9th Python in Science Conference (2010): 51-56. [https://pandas.pydata.org/](https://pandas.pydata.org/)
2. Pearu, Manoj et al. "tifffile: TIFF File I/O Library." (2010). [https://pypi.org/project/tifffile/](https://pypi.org/project/tifffile/)
3. Harris, CR et al. "Array programming with NumPy." Nature 520.7545 (2015): 315-322. [https://numpy.org/](https://numpy.org/)
4. Virtanen, Pauli et al. "SciPy 1.1--User Guide." (2020). [https://docs.scipy.org/doc/scipy/reference/ndimage.html](https://docs.scipy.org/doc/scipy/reference/ndimage.html) 



## Contact

For information about the paper contact Erika Sanchez-Aced esanchezac@santpau.cat
For information about the code contact Joaquim Aumatell Escabias jaumatelle@santpau.cat 
