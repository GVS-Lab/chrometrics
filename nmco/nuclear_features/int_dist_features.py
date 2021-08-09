# -*- coding: utf-8 -*-
"""
Library for computing features that describe the intensity distribution
This module provides functions that one can use to obtain and describe the intensity distribution of a given image

Available Functions:
-hetero_euchro_measures:Computes Heterochromatin to Euchromatin features
-intensity_histogram_measures: Computes descriptors of the intensity histogram
-measure_intensity_features: Computes all intensity distribution features
"""
# Import modules
import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew
from skimage import measure


def hetero_euchro_measures(regionmask: np.ndarray, intensity: np.ndarray, alpha: float = 1.0):
    """Computes Heterochromatin to Euchromatin features
    
    This functions obtains the Heterochromatin (high intensity) and Euchromatin (low intensity)
    and computes features that describe the relationship between the two
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        alpha     : threshold for calculating heterochromatin intensity
    """
    high, low = np.percentile(intensity[regionmask], q=(80, 20))
    hc = np.mean(intensity[regionmask]) + (alpha * np.std(intensity[regionmask]))

    feat = {
        "i80_i20": high / low,
        "nhigh_nlow": np.sum(intensity[regionmask] >= high)/ np.sum(intensity[regionmask] <= low),
        "hc_area_ec_area": np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] < hc),
        "hc_area_nuc_area": np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] > 0),
        "hc_content_ec_content": np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] < hc, intensity[regionmask], 0)),
        "hc_content_dna_content": np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] > 0, intensity[regionmask], 0))

    }
    return feat


def intensity_histogram_measures(regionmask: np.ndarray, intensity: np.ndarray):
    """Computes Intensity Distribution features
    
    This functions computes features that describe the distribution characteristic of the intensity.
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image

    """
    feat = {
        "int_min": np.percentile(intensity[regionmask], 0),
        "int_d25": np.percentile(intensity[regionmask], 25),
        "int_median": np.percentile(intensity[regionmask], 50),
        "int_d75": np.percentile(intensity[regionmask], 75),
        "int_max": np.percentile(intensity[regionmask], 100),
        "int_mean": np.mean(intensity[regionmask]),
        "int_mode": stats.mode(intensity[regionmask], axis=None)[0][0],
        "int_sd": np.std(intensity[regionmask]),
        "kurtosis": float(kurtosis(intensity[regionmask].ravel())),
        "skewness": float(skew(intensity[regionmask].ravel())),
        "entropy": shannon_entropy((intensity * regionmask)),

    }
    return feat


def measure_intensity_features(regionmask: np.ndarray, intensity: np.ndarray, measure_int_dist:bool = True, measure_hc_ec_ratios:bool = True, hc_alpha: int = 1):
    """Compute all intensity distribution features
    This function computes all features that describe the distribution of the gray levels. 
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    Returns: A pandas dataframe with all the features for the given image
    """

    feat ={}

    if(measure_int_dist):
        feat.update(intensity_histogram_measures(regionmask, intensity))
    if(measure_hc_ec_ratios):
        feat.update(hetero_euchro_measures(regionmask, intensity, hc_alpha))

    return pd.DataFrame([feat])
