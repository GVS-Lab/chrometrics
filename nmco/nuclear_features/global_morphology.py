# -*- coding: utf-8 -*-
"""
Library for computing features that describe the global boundary features

This module provides functions that one can use to obtain and describe the shape and size of an object

Available Functions:
-calliper_sizes:Obtains the min and max Calliper distances
-radii_features: Describing centroid to boundary distances
-simple_morphology: Complutes simple morphology features
-measure_global_morphometrics: Computes all global morphology features
"""

# import libraries
from skimage.morphology import erosion
from scipy import stats
from scipy.spatial import distance, distance_matrix
import numpy as np
from skimage.transform import rotate
import pandas as pd
from skimage import measure

def radii_features(binary_mask: np.ndarray):
    """Describing centroid to boundary distances(radii)
    This function obtains radii from the centroid to all the points along the edge and 
    using this computes features that describe the morphology of the given object.
    
    Args:
        binary_mask:(image_matrix) Background pixels are 0
    """
    
    foo = (measure.regionprops_table(binary_mask.astype(int),properties=["centroid"]))

    # obtain the edge pixels
    bw = binary_mask > 0
    cenx, ceny = (float(foo['centroid-0']),float(foo['centroid-1']))
    edge = np.subtract(bw * 1, erosion(bw) * 1)
    (boundary_x, boundary_y) = [np.where(edge > 0)[0], np.where(edge > 0)[1]]
   
    # calculate radii
    dist_b_c = np.sqrt(np.square(boundary_x - cenx) + np.square(boundary_y - ceny))
    cords = np.column_stack((boundary_x, boundary_y))
    dist_matrix = distance.squareform(distance.pdist(cords, "euclidean"))
    # Compute features
    feret = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]  # offset

    feat ={ "min_radius": np.min(dist_b_c),
            "max_radius": np.max(dist_b_c),
            "med_radius": np.median(dist_b_c),
            "avg_radius": np.mean(dist_b_c),
            "mode_radius": stats.mode(dist_b_c, axis=None).mode,
            "d25_radius": np.percentile(dist_b_c, 25),
            "d75_radius": np.percentile(dist_b_c, 75),
            "std_radius": np.std(dist_b_c),
            "feret_max": np.max(feret)
          }

    return feat


def calliper_sizes(binary_mask: np.ndarray, angular_resolution:int = 10):
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
        
    feat = { "min_calliper": min(callipers), 
             "max_calliper": max(callipers),
             "smallest_largest_calliper": min(callipers)/max(callipers)
           }
    
    return feat

def simple_morphology(regionmask: np.ndarray):
    """ Compute image moments
    Args:
        regionmask : binary background mask
    """
    
    morphology_features = ['centroid','area','perimeter','bbox_area','convex_area',
                            'equivalent_diameter','major_axis_length','minor_axis_length',
                            'eccentricity','orientation']
    
    regionmask=regionmask.astype('uint8')

    feat = pd.DataFrame(measure.regionprops_table(regionmask,properties=morphology_features))
    feat["concavity"] = (feat["convex_area"] - feat["area"]) / feat["convex_area"]
    feat["solidity"] = feat["area"] / feat["convex_area"]
    feat["a_r"] = feat["minor_axis_length"] / feat["major_axis_length"]
    feat["shape_factor"] = (feat["perimeter"] ** 2) / (4 * np.pi * feat["area"])
    feat["area_bbarea"] = feat["area"] / feat["bbox_area"]
    
    return feat

        
def measure_global_morphometrics(binary_image:np.ndarray, angular_resolution:int = 10, measure_simple:bool = True,
                                 measure_calliper:bool =True, measure_radii:bool = True):
    """Compute all boundary features
    This function computes all features that describe the boundary features
    Args:
        binary_image:(image_array) Binary image 
        angular_resolution:(integer) value between 1-359 to determine the number of rotations
    Returns: A pandas dataframe with all the features for the given image
    """

    feat ={}

    if(measure_calliper):
        feat.update(calliper_sizes(binary_image, angular_resolution))
    if(measure_radii):
        print("measuring radii")
        feat.update(radii_features(binary_image))
    if(measure_simple):
        feat = pd.concat([pd.DataFrame([feat]), simple_morphology(binary_image)], axis =1)
    else: 
        feat = pd.DataFrame([feat])
    return feat
