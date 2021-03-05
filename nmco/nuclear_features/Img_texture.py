# -*- coding: utf-8 -*-
"""
Library for computing features that describe the texture of a given image

This module provides functions that one can use to obtain and describe the texture of a given image

Available Functions:
-gclm_textures:Compute GLCM features at 1,5, 20 lengths
-texture_features: Computes all intensity distribution features
-Peripherial_Distribution_Index: Todo
"""
# Import modules
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte


def gclm_textures(regionmask, intensity, lengths=[1, 5, 20]):
    """ Compute GLCM features at given lengths
    
    Args:
        regionmask=binary image
        intensity= intensity image
    """
    # Contruct GCL matrix at given pixels lengths
    glcm = greycomatrix(
        img_as_ubyte((intensity * regionmask) / 255),
        distances=lengths,
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    )
    contrast = pd.DataFrame(np.mean(greycoprops(glcm, "contrast"), axis=1).tolist()).T
    contrast.columns = ["Contrast_" + str(col) for col in lengths]
    dissimilarity = pd.DataFrame(
        np.mean(greycoprops(glcm, "dissimilarity"), axis=1).tolist()
    ).T
    dissimilarity.columns = ["dissimilarity_" + str(col) for col in lengths]
    homogeneity = pd.DataFrame(
        np.mean(greycoprops(glcm, "homogeneity"), axis=1).tolist()
    ).T
    homogeneity.columns = ["homogeneity_" + str(col) for col in lengths]
    ASM = pd.DataFrame(np.mean(greycoprops(glcm, "ASM"), axis=1).tolist()).T
    ASM.columns = ["ASM_" + str(col) for col in lengths]
    energy = pd.DataFrame(np.mean(greycoprops(glcm, "energy"), axis=1).tolist()).T
    energy.columns = ["energy_" + str(col) for col in lengths]
    correlation = pd.DataFrame(
        np.mean(greycoprops(glcm, "correlation"), axis=1).tolist()
    ).T
    correlation.columns = ["correlation_" + str(col) for col in lengths]

    feat = pd.concat(
        [
            contrast.reset_index(drop=True),
            dissimilarity.reset_index(drop=True),
            homogeneity.reset_index(drop=True),
            ASM.reset_index(drop=True),
            energy.reset_index(drop=True),
            correlation.reset_index(drop=True),
        ],
        axis=1,
    )

    return feat


def Peripherial_Distribution_Index(regionmask, intensity):
    """Computes pheriphal distribution index of a grayscale image
    """
    pass


def texture_features(regionmask, intensity, lengths=[1, 5, 20]):
    """Compute all texture features
    This function computes all features that describe the image texture 
    Args:
        regionmask=binary image
        intensity= intensity image
    Returns: A pandas dataframe with all the features for the given image
    """

    # compute features
    gclm_measures = gclm_textures(regionmask, intensity, lengths)

    return gclm_measures
