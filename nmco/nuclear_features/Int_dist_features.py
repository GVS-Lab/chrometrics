# -*- coding: utf-8 -*-
"""
Library for computing features that describe the intensity distribution

This module provides functions that one can use to obtain and describe the intensity distribution of a given image

Available Functions:
-hetero_euchro_measures:Computes Heterochromatin to Euchromatin features
-intensity_histogram_measures: Computes Intensity histogram features
-entropy_image(regionmask, intensity): Compute Shannon Entropy of a given image
-intensity_features: Computes all intensity distribution features
"""
# Import modules
import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import shannon_entropy


class Hetero_Euchro_Measures:
    def __init__(self, output_hetero_euchro_measures):
        self.i80_i20 = output_hetero_euchro_measures[0]
        self.nhigh_nlow = output_hetero_euchro_measures[1]
        self.hc_area_ec_area = output_hetero_euchro_measures[2]
        self.hc_area_nuc_area = output_hetero_euchro_measures[3]
        self.hc_content_ec_content = output_hetero_euchro_measures[4]
        self.hc_content_dna_content = output_hetero_euchro_measures[5]


def hetero_euchro_measures(regionmask, intensity):
    """Computes Heterochromatin to Euchromatin features
    
    This functions obtains the Heterochromatin (high intensity) and Euchromatin (low intensity)
    and computes features that describe the relationship between the two
    
    Args:
        regionmask=binary image
        intensity= intensity image
    """
    high, low = np.percentile(intensity[regionmask], q=(80, 20))
    hc = np.mean(intensity[regionmask]) + (1.5 * np.std(intensity[regionmask]))
    feat = Hetero_Euchro_Measures(
        [
            high / low,
            np.sum(intensity[regionmask] >= high)
            / np.sum(intensity[regionmask] <= low),
            np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] < hc),
            np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] > 0),
            np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] < hc, intensity[regionmask], 0)),
            np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] > 0, intensity[regionmask], 0)),
        ]
    )
    return feat


class Intensity_Histogram_Measures:
    def __init__(self, output_intensity_histogram_measures):
        self.int_min = output_intensity_histogram_measures[0]
        self.int_d25 = output_intensity_histogram_measures[1]
        self.int_median = output_intensity_histogram_measures[2]
        self.int_d75 = output_intensity_histogram_measures[3]
        self.int_max = output_intensity_histogram_measures[4]
        self.int_mean = output_intensity_histogram_measures[5]
        self.int_mode = output_intensity_histogram_measures[6]
        self.int_sd = output_intensity_histogram_measures[7]


def intensity_histogram_measures(regionmask, intensity):
    """Computes Intensity Distribution features
    
    This functions computes features that describe the distribution characteristic of the instensity.
    Args:
        regionmask=binary image
        intensity= intensity image
    """
    feat = Intensity_Histogram_Measures(
        [
            np.percentile(intensity[regionmask], 0),
            np.percentile(intensity[regionmask], 25),
            np.percentile(intensity[regionmask], 50),
            np.percentile(intensity[regionmask], 75),
            np.percentile(intensity[regionmask], 100),
            np.mean(intensity[regionmask]),
            stats.mode(intensity[regionmask], axis=None)[0][0],
            np.std(intensity[regionmask]),
        ]
    )
    return feat


class Entropy_Image:
    def __init__(self, output_entropy_image):
        self.entropy = output_entropy_image[0]


def entropy_image(regionmask, intensity):
    """Compute Shannon Entropy of a given image
    
    Args:
        regionmask=binary image
        intensity= intensity image
    """
    feat = Entropy_Image([shannon_entropy((intensity * regionmask))])
    return feat


def intensity_features(regionmask, intensity):
    """Compute all intensity distribution features
    This function computes all features that describe the distribution of the gray levels. 
    Args:
        regionmask=binary image
        intensity= intensity image
    Returns: A pandas dataframe with all the features for the given image
    """

    # compute features
    dist_measures = [intensity_histogram_measures(regionmask, intensity)]
    dist_measures = pd.DataFrame([o.__dict__ for o in dist_measures])

    he_feat = [hetero_euchro_measures(regionmask, intensity)]
    he_feat = pd.DataFrame([o.__dict__ for o in he_feat])

    entropy_feat = [entropy_image(regionmask, intensity)]
    entropy_feat = pd.DataFrame([o.__dict__ for o in entropy_feat])

    all_features = pd.concat([entropy_feat.reset_index(drop=True), he_feat], axis=1)
    all_features = pd.concat(
        [dist_measures.reset_index(drop=True), all_features], axis=1
    )

    return all_features
