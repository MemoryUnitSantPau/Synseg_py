# FUNCTIONS SUMMARY

## Introduction

This repository contains the scripts used to perorm Segmentation, Density and Distnce analysis of proteins in STORM images.

## apply_threshold_and_binarize
Given an image and a percentage, this function applies thresholding , selecting the data with values over the percentage of the maximum intenity of the image, and binarization.

## group_files_by_common_part
Groups filenames by their common parts and sorts the groups based on neuropil_channel presence.

## calculate_per_dist
Calculate the minimum Euclidean distance between two sets of points.

## initialize_alignment
Initialize the alignment process with image data and optional presets.

## check_continuity
Given a list count the consecutive elements containing 1 or true.

## cmyk_to_rgb
Convert CMYK color values to RGB color values.

## modify_filename
Modify a filename to make it unique if it already exists in a specified directory.

## combine_and_draw2
Combine two images and draw a line between two specified points.

## object_identificator
Identify and analyze objects in a preprocessed image frame.

## combine_and_draw
Combine two images and draw a colored line between two points using Bresenham's line algorithm.

## preprocess_image
Preprocesses a single frame of an image for further analysis.

## combine_plots
Combine grayscale images and plot them in a 1x3 grid.

## extract_common_part2
Group a list of filenames by their common parts based on unique texts.

## roi_check
Display a frame, its corresponding ROI, and the template side by side.

## extract_common_part
Extracts the common part of a filename up to the first underscore.

## ROI_MAP
Generate a map of regions of interest (ROIs) in image frames.

## extract_unique_texts_between_parentheses
Extract unique texts found between parentheses in a list of strings.

## save_results
Save various results including CSV files, aligned images, and visualizations.

## extract_unique_texts_between_underscores
Extracts unique texts found between underscores in a list of strings.

## spiderwebs
Generate spiderweb-like visualizations from image data.

## update_template
Update a template image with a selected ROI and store ROI information in a DataFrame.

## group_files_by_common_part2
Group a list of filenames by their common parts based on unique texts

## wrapper_function
