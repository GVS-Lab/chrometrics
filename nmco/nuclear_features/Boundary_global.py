# -*- coding: utf-8 -*-
"""
Library for computing features that describe the global boundary features

This module provides functions that one can use to obtain and describe the shape and size of an object

Available Functions:
-calliper_sizes:Obtains the min and max Calliper distances
-radii_features: Describing centroid to boundary distances
-boundary_features: Computes all curvature features
"""

# import libraries
from skimage.morphology import erosion
from scipy import stats
from scipy.spatial import distance, distance_matrix
import numpy as np
from skimage.transform import rotate
import pandas as pd


class Radii_Features:
    def __init__(self, output_radii_features):
        self.min_radius = output_radii_features[0]
        self.max_radius = output_radii_features[1]
        self.med_radius = output_radii_features[2]
        self.avg_radius = output_radii_features[3]
        self.mode_radius = output_radii_features[4]
        self.d25_radius = output_radii_features[5]
        self.d75_radius = output_radii_features[6]
        self.std_radius = output_radii_features[7]
        self.feret_max = output_radii_features[8]


class Calliper_Features:
    def __init__(self, output_calliper_sizes):
        self.min_calliper = output_calliper_sizes[0]
        self.max_calliper = output_calliper_sizes[1]


def radii_features(binary_mask, centroid):
    """Describing centroid to boundary distances(radii)
    This function obtains radii from the centroid to all the points along the edge and 
    using this computes features that describe the morphology of the given object.
    
    Args:
        binary_mask:(image_matrix) Background pixels are 0
        centroid: (tuple: x, y coordinates) local centroid of the object
    """
    # obtain the edge pixels
    bw = binary_mask > 0
    cenx, ceny = centroid
    edge = np.subtract(bw * 1, erosion(bw) * 1)
    (boundary_x, boundary_y) = [np.where(edge > 0)[0], np.where(edge > 0)[1]]
    # calculate radii
    dist_b_c = np.sqrt(np.square(boundary_x - cenx) + np.square(boundary_y - ceny))
    cords = np.column_stack((boundary_x, boundary_y))
    dist_matrix = distance.squareform(distance.pdist(cords, "euclidean"))
    # Compute features
    feret = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]  # offset

    feat = Radii_Features(
        [
            np.min(dist_b_c),
            np.max(dist_b_c),
            np.median(dist_b_c),
            np.mean(dist_b_c),
            stats.mode(dist_b_c, axis=None)[0][0],
            np.percentile(dist_b_c, 25),
            np.percentile(dist_b_c, 75),
            np.std(dist_b_c),
            np.max(feret),
        ]
    )

    return feat


def calliper_sizes(binary_mask, angular_resolution=10):
    """Obtains the min and max Calliper distances
    
    This functions calculates min and max the calliper distances by rotating the image
    by the given angular resolution
    
    Args: 
        binary_mask:(image_arrray)
        angular_resolution:(integer) value between 1-359 to determine the number of rotations
        
    """
    img = binary_mask > 0
    callipers = []
    for angle in range(1, 360, 10):
        rot_img = rotate(img, angle, resize=True)
        callipers.append(max(np.sum(rot_img, axis=0)))
    feat = Calliper_Features([min(callipers), max(callipers)])
    return feat


def boundary_features(
    binary_image, centroids, angular_resolution=10,
):
    """Compute all boundary features
    This function computes all features that describe the boundary features
    Args:
        binary_image:(image_array) Binary image 
        angular_resolution:(integer) value between 1-359 to determine the number of rotations
        centroids:(tuple: x, y coordinates) local centroid of the object
    Returns: A pandas dataframe with all the features for the given image
    """

    # compute local and global features
    calliper_features = [calliper_sizes(binary_image)]
    calliper_features = pd.DataFrame([o.__dict__ for o in calliper_features])

    radii_parameters = [radii_features(binary_image, centroids)]
    radii_parameters = pd.DataFrame([o.__dict__ for o in radii_parameters])

    all_features = pd.concat(
        [calliper_features.reset_index(drop=True), radii_parameters], axis=1
    )

    return all_features
