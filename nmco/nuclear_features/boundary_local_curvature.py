# -*- coding: utf-8 -*-
"""
Library for computing features that describe the local boundary curvature

This module provides functions that one can use to obtain, visualise and describe local curvature of a given object. 

Available Functions:
-circumradius:Finds the radius of a circumcircle
-local_radius_curvature: Computes local radius of curvature
-global_curvature_features: Obtain features describing an object's local curvatures
-prominant_curvature_features: Obtain prominant (peaks) local curvature
-measure_curvature_features: Computes all curvature features
"""
# Import libraries
import numpy as np
import pandas as pd
from math import degrees, sqrt
import matplotlib.pyplot as plt
from skimage.morphology import erosion
from scipy import signal
from skimage import measure


def circumradius(T, binary_mask: np.ndarray):
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


def local_radius_curvature(binary_image: np.ndarray, step:int = 2, show_boundary:bool =False):
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

def global_curvature_features(local_curvatures: np.ndarray):
    """Obtain features describing an object's local curvatures
    This function computres features that describe the local curvature distributions, 
    Args:
        local_curvatures:(Array) of ordered local curvatures
         
    """

    # differentiate postive and negative curvatures and compute features
    pos_curvature = local_curvatures[local_curvatures > 0]
    neg_curvature = np.abs(local_curvatures[local_curvatures < 0])

    feat = {"avg_curvature" : np.mean(local_curvatures),
            "std_curvature" :  np.std(local_curvatures),
            "npolarity_changes" : np.where(np.diff(np.sign(local_curvatures)))[0].shape[0]
           }
                    
    if pos_curvature.shape[0] > 0:
        positive_feat={"max_posi_curv": np.max(pos_curvature),
                       "avg_posi_curv": np.mean(pos_curvature),
                       "med_posi_curv": np.median(pos_curvature),
                       "std_posi_curv": np.std(pos_curvature),
                       "sum_posi_curv": np.sum(pos_curvature),
                       "len_posi_curv": pos_curvature.shape[0]
        }
    else:
        positive_feat={"max_posi_curv": np.nan,
                       "avg_posi_curv": np.nan,
                       "med_posi_curv": np.nan,
                       "std_posi_curv": np.nan,
                       "sum_posi_curv": np.nan,
                       "len_posi_curv": np.nan
        }
    
    feat.update(positive_feat)

    if neg_curvature.shape[0] > 0:
        negative_feat={"max_neg_curv": np.max(neg_curvature),
                       "avg_neg_curv": np.mean(neg_curvature),
                       "med_neg_curv": np.median(neg_curvature),
                       "std_neg_curv": np.std(neg_curvature),
                       "sum_neg_curv": np.sum(neg_curvature),
                       "len_neg_curv": neg_curvature.shape[0]
                      }

    else:
        negative_feat={"max_neg_curv": np.nan,
                       "avg_neg_curv": np.nan,
                       "med_neg_curv": np.nan,
                       "std_neg_curv": np.nan,
                       "sum_neg_curv": np.nan,
                       "len_neg_curv": np.nan
                      }

    
    feat.update(negative_feat)        
    return feat

def prominant_curvature_features(
    local_curvatures:np.ndarray,
    show_plot:bool=False,
    min_prominance:float=0.1,
    min_width:int=5,
    dist_bwt_peaks:int=10,
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
        prominance_prominant_positive_curvature = np.nan
        width_prominant_positive_curvature = np.nan
        prominant_positive_curvature = np.nan

    num_prominant_negative_curvature = len(neg_peaks)
    if len(neg_peaks) > 0:
        prominance_prominant_negative_curvature = np.mean(neg_prop["prominences"])
        width_prominant_negative_curvature = np.mean(neg_prop["widths"])
        prominant_negative_curvature = np.mean(
            [local_curvatures[neg_peaks[x]] for x in range(len(neg_peaks))]
        )
    elif len(neg_peaks) == 0:
        prominance_prominant_negative_curvature = np.nan
        width_prominant_negative_curvature = np.nan
        prominant_negative_curvature = np.nan

    feat = { "num_prominant_pos_curv" : num_prominant_positive_curvature,
             "prominance_prominant_pos_curv" : prominance_prominant_positive_curvature,
             "width_prominant_pos_curv" : width_prominant_positive_curvature,
             "prominant_pos_curv": prominant_positive_curvature,
             "num_prominant_neg_curv" : num_prominant_negative_curvature,
             "prominance_prominant_neg_curv" : prominance_prominant_negative_curvature,
             "width_prominant_neg_curv" : width_prominant_negative_curvature,
             "prominant_neg_curv": prominant_negative_curvature,
            
    }
    
    return feat


def measure_curvature_features(
    binary_image:np.ndarray, step:int = 2, prominance:float = 0.1, width:int = 5, dist_bt_peaks:int =10):
    """Comupte all curvature features
    This function computes all features that describe the local boundary features
    
    Args:
        binary_image:(image_array) Binary image 
        step: (integer) Step size used to obtain the vertices, use larger values for a smoother curvatures
        prominance: (numeric) minimal required prominance of peaks (Default=0.1)
        width: (numeric) minimum width required of peaks (Deafult=5)
        dist_bt_peaks: (numeric) required minimum distance between peaks (Default=10)
    
    Returns: A pandas dataframe with all the features for the given image
    """
    r_c = local_radius_curvature(binary_image, step, False)

    # calculate local curvature features
    local_curvature = np.array([
        np.divide(1, r_c[x]) if r_c[x] != 0 else 0 for x in range(len(r_c))
    ])
    feat ={}
    feat.update(global_curvature_features(local_curvatures = local_curvature))
    feat.update(prominant_curvature_features(local_curvatures = local_curvature,
                                             min_prominance = prominance,
                                             min_width = width,
                                             dist_bwt_peaks = dist_bt_peaks))
    feat = pd.DataFrame([feat])
    foo = (measure.regionprops_table(binary_image.astype(int),properties=["perimeter"]))
    perimeter = float(foo["perimeter"])
    
    feat["frac_peri_w_posi_curvature"] = (feat["len_posi_curv"].replace(to_replace="NA", value=0)/ perimeter)
    feat["frac_peri_w_neg_curvature"] = (feat["len_neg_curv"].replace(to_replace="NA", value=0)/perimeter)
    feat["frac_peri_w_polarity_changes"] = (feat["npolarity_changes"] / perimeter)
 
    return feat
