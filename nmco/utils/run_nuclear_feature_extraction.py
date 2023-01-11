# -*- coding: utf-8 -*-
from tifffile import imread
import pandas as pd
from skimage import measure
import numpy as np
import cv2 as cv
from nmco.nuclear_features import (
    global_morphology as BG,
    img_texture as IT,
    int_dist_features as IDF,
    boundary_local_curvature as BLC
)
from tqdm import tqdm

def run_nuclear_chromatin_feat_ext(raw_image_path:str, labelled_image_path:str, output_dir:str,
                                   calliper_angular_resolution:int = 10, 
                                   measure_simple_geometry:bool = True, 
                                   measure_calliper_distances:bool = True, 
                                   measure_radii_features:bool = True,
                                   step_size_curvature:int = 2, 
                                   prominance_curvature:float = 0.1, 
                                   width_prominent_curvature:int = 5, 
                                   dist_bt_peaks_curvature:int = 10,
                                   measure_int_dist_features:bool = True, 
                                   measure_hc_ec_ratios_features:bool = True, 
                                   hc_threshold:float = 1, 
                                   gclm_lengths:list = [1, 5, 20],
                                   measure_gclm_features: bool = True, 
                                   measure_moments_features: bool = True,
                                   normalize:bool=False, 
                                   save_output:bool = False):
    """
    Function that reads in the raw and segmented/labelled images for a field of view and computes nuclear features. 
    Note this has been used only for DAPI stained images
    Args:
        raw_image_path: path pointing to the raw image
        labelled_image_path: path pointing to the segmented image
        output_dir: path where the results need to be stored
    """
    labelled_image = imread(labelled_image_path)
    raw_image = imread(raw_image_path)
    labelled_image = labelled_image.astype(int)
    raw_image = raw_image.astype(int)

    # Insert code for preprocessing image
    # Eg normalize
    if normalize:
        raw_image = cv.normalize(
         raw_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
     )
        raw_image[raw_image < 0] = 0.0
        raw_image[raw_image > 255] = 255.0

    # Get features for the individual nuclei in the image
    props = measure.regionprops(labelled_image, raw_image)
    
    all_features = pd.DataFrame()
    # Measure scikit's built in features
    
    for i in tqdm(range(len(props))):
        all_features = all_features.append(
            pd.concat(
                [pd.DataFrame([props[i].label], columns=["label"]),
                 BG.measure_global_morphometrics(props[i].image, 
                                                 angular_resolution = calliper_angular_resolution, 
                                                 measure_simple = measure_simple_geometry,
                                                 measure_calliper = measure_calliper_distances, 
                                                 measure_radii = measure_radii_features).reset_index(drop=True),
                 BLC.measure_curvature_features(props[i].image, step = step_size_curvature, 
                                                prominance = prominance_curvature, 
                                                width = width_prominent_curvature, 
                                                dist_bt_peaks = dist_bt_peaks_curvature).reset_index(drop=True),
                 IDF.measure_intensity_features(props[i].image, props[i].intensity_image, 
                                                measure_int_dist = measure_int_dist_features, 
                                                measure_hc_ec_ratios = measure_hc_ec_ratios_features, 
                                                hc_alpha = hc_threshold).reset_index(drop=True),
                 IT.measure_texture_features(props[i].image, props[i].intensity_image, lengths=gclm_lengths,
                                             measure_gclm = measure_gclm_features,
                                             measure_moments = measure_moments_features)],
                axis=1,
            ),
            ignore_index=True,
        )
   
    #save the output
    if save_output:
        all_features.to_csv(output_dir+"/"+labelled_image_path.rsplit('/', 1)[-1][:-4]+".csv")

    return all_features
