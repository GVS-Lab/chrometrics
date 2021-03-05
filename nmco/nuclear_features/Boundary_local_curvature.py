# -*- coding: utf-8 -*-
"""
Library for computing features that describe the local boundary curvature

This module provides functions that one can use to obtain, visualise and describe local curvature of a given object. 

Available Functions:
-circumradius:Finds the radius of a circumcircle
-local_radius_curvature: Computes local radius of curvature
-global_curvature_features: Obtain features describing an object's local curvatures
-prominant_curvature_features: Obtain prominant (peaks) local curvature
-curvature_features: Computes all curvature features
"""
# Import libraries
import numpy as np
import pandas as pd
from math import degrees, sqrt
import matplotlib.pyplot as plt
from skimage.morphology import erosion
from scipy import signal


def circumradius(T, binary_mask):
    """Finds the radius of a circumcircle
    This functions calculates the radius of circumcircle given the coordinates of 3 points. 
    The sign of the radius is positive if the circumcenter is inside the binary image.
    
    Args: 
        T: (tuple) cartesian coordinatesof three points:(x1, y1), (x2, y2), (x3, y3)
        binary_mask: (image_matrix) The foreground pixels have a value of one. 
    
    Returns:
        Radius of the circumcircle. False if it cannot be calculated eg: if the points are colinear.  
    """
    (x1, y1), (x2, y2), (x3, y3) = T  # extracting the points.

    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))  # Diameter
    if D != 0:
        # Centroid of the cicumcircle
        Ux = (
            ((x1 ** 2 + y1 ** 2) * (y2 - y3))
            + ((x2 ** 2 + y2 ** 2) * (y3 - y1))
            + ((x3 ** 2 + y3 ** 2) * (y1 - y2))
        ) / D
        Uy = (
            ((x1 ** 2 + y1 ** 2) * (x3 - x2))
            + ((x2 ** 2 + y2 ** 2) * (x1 - x3))
            + ((x3 ** 2 + y3 ** 2) * (x2 - x1))
        ) / D

        # radius
        r = sqrt((Ux - x2) ** 2 + (Uy - y2) ** 2)
        r = r + 1

        # Determining the sign: it is positive if the centroid of the circumcricle is in the foreground
        x = np.floor(Ux).astype(int)
        y = np.floor(Uy).astype(int)

        if x >= binary_mask.shape[0] or y >= binary_mask.shape[1]:
            r = -r
        elif x < 0 or y < 0:
            r = -r
        elif binary_mask[x, y]:
            r = r
        else:
            r = -r
        return r
    else:
        return False


def local_radius_curvature(binary_image, step=2, show_boundary=False):
    """Computes local radius of curvature. 
    
    This functions calculates the local curvatures given a segmented image.  
    
    Args: 
        binary_image: (image_matrix) binary region image of a segmented object.
        step: (integer) Step size used to obtain the vertices, use larger values for a smoother curvatures
        show_boundary: (Logical) true if function should plot raduis of curvature 
    Returns:
        List of local curvature features
          
    """

    # obtain the edge of the given binary image.
    bw = binary_image > 0
    bw = np.pad(bw, pad_width=5, mode="constant", constant_values=0)
    edge = np.subtract(bw * 1, erosion(bw) * 1)
    (boundary_x, boundary_y) = [np.where(edge > 0)[0], np.where(edge > 0)[1]]
    cenx, ceny = np.mean(boundary_x), np.mean(boundary_y)
    arr1inds = np.arctan2(boundary_x - cenx, boundary_y - ceny).argsort()
    boundary_x, boundary_y = boundary_x[arr1inds[::-1]], boundary_y[arr1inds[::-1]]

    # obtain local radii of curvature with the given step size
    cords = np.column_stack((boundary_x, boundary_y))
    cords_circ = np.vstack((cords[-step:], cords, cords[:step]))
    r_c = np.array(
        [
            circumradius(
                (cords_circ[i - step], cords_circ[i], cords_circ[i + step]), bw
            )
            for i in range(step, cords.shape[0] + step)
        ]
    )

    # plot an image of the boundary with the curvature if asked
    if show_boundary:
        edge[boundary_x, boundary_y] = r_c
        plt.imshow(edge)
        plt.colorbar()
    return r_c


class Global_Curvature_Features:
    def __init__(self, output_global_curvature_features):
        self.avg_curvature = output_global_curvature_features[0]
        self.atd_curvature = output_global_curvature_features[1]
        self.npolarity_changes = output_global_curvature_features[2]
        self.max_posi_curvature = output_global_curvature_features[3]
        self.mvg_posi_curvature = output_global_curvature_features[4]
        self.med_posi_curvature = output_global_curvature_features[5]
        self.std_posi_curvature = output_global_curvature_features[6]
        self.sum_posi_curvature = output_global_curvature_features[7]
        self.len_posi_curvature = output_global_curvature_features[8]
        self.max_neg_curvature = output_global_curvature_features[9]
        self.avg_neg_curvature = output_global_curvature_features[10]
        self.med_neg_curvature = output_global_curvature_features[11]
        self.std_neg_curvature = output_global_curvature_features[12]
        self.sum_neg_curvature = output_global_curvature_features[13]
        self.len_neg_curvature = output_global_curvature_features[14]


def global_curvature_features(local_curvatures):
    """Obtain features describing an object's local curvatures
    This function computres features that describe the local curvature distributions, 
    Args:
        local_curvatures:(Array) of ordered local curvatures
       
    Returns: Object with the values
       
    """

    # differentiate postive and negative curvatures and compute features
    pos_curvature = local_curvatures[local_curvatures > 0]
    neg_curvature = np.abs(local_curvatures[local_curvatures < 0])

    if pos_curvature.shape[0] > 0:
        (
            max_posi_curv,
            avg_posi_curv,
            med_posi_curv,
            std_posi_curv,
            sum_posi_curv,
            len_posi_curv,
        ) = (
            np.max(pos_curvature),
            np.mean(pos_curvature),
            np.median(pos_curvature),
            np.std(pos_curvature),
            np.sum(pos_curvature),
            pos_curvature.shape[0],
        )
    else:
        (
            max_posi_curv,
            avg_posi_curv,
            med_posi_curv,
            std_posi_curv,
            sum_posi_curv,
            len_posi_curv,
        ) = ("NA", "NA", "NA", "NA", "NA", "NA")

    if neg_curvature.shape[0] > 0:
        (
            max_neg_curv,
            avg_neg_curv,
            med_neg_curv,
            std_neg_curv,
            sum_neg_curv,
            len_neg_curv,
        ) = (
            np.max(neg_curvature),
            np.mean(neg_curvature),
            np.median(neg_curvature),
            np.std(neg_curvature),
            np.sum(neg_curvature),
            neg_curvature.shape[0],
        )

    else:
        (
            max_neg_curv,
            avg_neg_curv,
            med_neg_curv,
            std_neg_curv,
            sum_neg_curv,
            len_neg_curv,
        ) = ("NA", "NA", "NA", "NA", "NA", "NA")

    return Global_Curvature_Features(
        [
            np.mean(local_curvatures),
            np.std(local_curvatures),
            np.where(np.diff(np.sign(local_curvatures)))[0].shape[0],
            max_posi_curv,
            avg_posi_curv,
            med_posi_curv,
            std_posi_curv,
            sum_posi_curv,
            len_posi_curv,
            max_neg_curv,
            avg_neg_curv,
            med_neg_curv,
            std_neg_curv,
            sum_neg_curv,
            len_neg_curv,
        ]
    )


class Prominant_Curvature_Features:
    def __init__(self, output_prominant_curvature_features):
        self.num_prominant_positive_curvature = output_prominant_curvature_features[0]
        self.prominance_prominant_positive_curvature = output_prominant_curvature_features[
            1
        ]
        self.width_prominant_positive_curvature = output_prominant_curvature_features[2]
        self.prominant_positive_curvature = output_prominant_curvature_features[3]
        self.num_prominant_negative_curvature = output_prominant_curvature_features[4]
        self.prominance_prominant_negative_curvature = output_prominant_curvature_features[
            5
        ]
        self.width_prominant_negative_curvature = output_prominant_curvature_features[6]
        self.prominant_negative_curvature = output_prominant_curvature_features[6]


def prominant_curvature_features(
    local_curvatures,
    show_plot=False,
    min_prominance=0.1,
    min_width=5,
    dist_bwt_peaks=10,
):
    """Obtain prominant (peaks) local curvature
    This function finds peaks for a given list of local curvatures using scipy's signal module.  

    Args:
        local_curvatures:(Array) of ordered local curvatures
        show_plot: (logical) true if the function should plot the identified peaks
        min_prominance: (numeric) minimal required prominance of peaks (Default=0.1)
        min_width: (numeric) minimum width required of peaks (Deafult=5)
        dist_bwt_peaks: (numeric) required minimum distance between peaks (Default=10)
    
    Returns: Object with the values: 
        num_prominant_positive_curvature,
        prominance_prominant_positive_curvature,
        width_prominant_positive_curvature,
        prominant_positive_curvature,
        num_prominant_negative_curvature,
        prominance_prominant_negative_curvature,
        width_prominant_negative_curvature,
        prominant_negative_curvature
    """
    # Find positive and nevative peaks
    pos_peaks, pos_prop = signal.find_peaks(
        local_curvatures,
        prominence=min_prominance,
        distance=dist_bwt_peaks,
        width=min_width,
    )
    neg_peaks, neg_prop = signal.find_peaks(
        [local_curvatures[x] * -1 for x in range(len(local_curvatures))],
        prominence=min_prominance,
        distance=dist_bwt_peaks,
        width=min_width,
    )

    # if specified show plot
    if show_plot:
        plt.plot(np.array(local_curvatures))
        plt.plot(pos_peaks, np.array(local_curvatures)[pos_peaks], "x")
        plt.plot(neg_peaks, np.array(local_curvatures)[neg_peaks], "x")
        plt.ylabel = "Curvature"
        plt.xlabel = "Boundary"
    # compute features
    num_prominant_positive_curvature = len(pos_peaks)
    if len(pos_peaks) > 0:
        prominance_prominant_positive_curvature = np.mean(pos_prop["prominences"])
        width_prominant_positive_curvature = np.mean(pos_prop["widths"])
        prominant_positive_curvature = np.mean(
            [local_curvatures[pos_peaks[x]] for x in range(len(pos_peaks))]
        )
    elif len(pos_peaks) == 0:
        prominance_prominant_positive_curvature = "NA"
        width_prominant_positive_curvature = "NA"
        prominant_positive_curvature = "NA"

    num_prominant_negative_curvature = len(neg_peaks)
    if len(neg_peaks) > 0:
        prominance_prominant_negative_curvature = np.mean(neg_prop["prominences"])
        width_prominant_negative_curvature = np.mean(neg_prop["widths"])
        prominant_negative_curvature = np.mean(
            [local_curvatures[neg_peaks[x]] for x in range(len(neg_peaks))]
        )
    elif len(neg_peaks) == 0:
        prominance_prominant_negative_curvature = "NA"
        width_prominant_negative_curvature = "NA"
        prominant_negative_curvature = "NA"

    return Prominant_Curvature_Features(
        [
            num_prominant_positive_curvature,
            prominance_prominant_positive_curvature,
            width_prominant_positive_curvature,
            prominant_positive_curvature,
            num_prominant_negative_curvature,
            prominance_prominant_negative_curvature,
            width_prominant_negative_curvature,
            prominant_negative_curvature,
        ]
    )


def curvature_features(
    binary_image, step=2, min_prominance=0.1, min_width=5, dist_bwt_peaks=10
):
    """Comupte all curvature features
    This function computes all features that describe the local boundary features
    
    Args:
        binary_image:(image_array) Binary image 
        step: (integer) Step size used to obtain the vertices, use larger values for a smoother curvatures
        min_prominance: (numeric) minimal required prominance of peaks (Default=0.1)
        min_width: (numeric) minimum width required of peaks (Deafult=5)
        dist_bwt_peaks: (numeric) required minimum distance between peaks (Default=10)
    
    Returns: A pandas dataframe with all the features for the given image
    """
    r_c = local_radius_curvature(binary_image, step, False)

    # calculate local curvature features
    local_curvature = [
        np.divide(1, r_c[x]) if r_c[x] != 0 else 0 for x in range(len(r_c))
    ]

    # compute local and global features
    global_features = [global_curvature_features(np.array(local_curvature))]
    global_features = pd.DataFrame([o.__dict__ for o in global_features])

    prominant_features = [
        prominant_curvature_features(
            local_curvature, min_prominance=0.1, min_width=5, dist_bwt_peaks=10
        )
    ]
    prominant_features = pd.DataFrame([o.__dict__ for o in prominant_features])

    all_features = pd.concat(
        [global_features.reset_index(drop=True), prominant_features], axis=1
    )

    return all_features
