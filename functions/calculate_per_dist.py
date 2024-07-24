import numpy as np
from math import sqrt

def calculate_per_dist(POS_A:list, 
                       POS_B:list,
                       elements:int = 5)->list:
    """
    Calculate the minimum Euclidean distance between two sets of points.

    Parameters:
        POS_A (list of tuples): List of (X, Y) coordinates for points in set A.
        POS_B (list of tuples): List of (X, Y) coordinates for points in set B.

    Returns:
        list: A list containing the following elements:
            - The minimum Euclidean distance between any pair of points.
            - The (X, Y) coordinates of the nearest point in set A.
            - The (X, Y) coordinates of the nearest point in set B.

    Example:
        POS_A = [(1, 2), (3, 4), (5, 6)]
        POS_B = [(2, 2), (4, 4), (7, 7)]
        result = calculate_per_dist(POS_A, POS_B)
        # Output: [1.0, (1, 2), (2, 2)]

    Note:
        - If only one object is found in either set A or B, the minimum distance will be "ONLY 1 OBJECT FOUND".
    """
    Distances = []
    A_points = []
    B_points = []
    for PA in POS_A:
        XA, YA = PA[0], PA[1]
        for PB in POS_B:
            XB, YB = PB[0], PB[1]
            VAR_X = XA - XB if XA > XB else XB - XA
            VAR_Y = YA - YB if  YA > YB else YB - YA
            A_points.append([XA,YA])
            B_points.append([XB, YB])
            Distances.append(sqrt(VAR_X**2 + VAR_Y**2))
    
    avg_min_dist = sum(sorted(Distances, key=lambda x: x)[:elements])/elements
    output = [min(Distances) if len(Distances) > 0 else "ONLY 1 OBJECT FOUND"]
    min_distance_indices = np.where(Distances == np.min(Distances))[0][0]
    output.append(A_points[min_distance_indices])
    output.append(B_points[min_distance_indices])
    output.append(avg_min_dist)
    return output