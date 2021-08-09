# -*- coding: utf-8 -*-
"""
Library for computing features that describe the texture of a given image
This module provides functions that one can use to obtain and describe the texture of a given image
Available Functions:
-gclm_textures:Compute GLCM features at different length sclates
-Peripherial_Distribution_Index: Todo
"""

# Import modules
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte
from skimage import measure


def gclm_textures(regionmask: np.ndarray, intensity: np.ndarray, lengths=[1, 5, 20]):
    """ Compute GLCM features at given lengths
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales 
     """
    # Contruct GCL matrix at given pixels lengths

    glcm = greycomatrix(
        img_as_ubyte((intensity * regionmask) / 255),
        distances=lengths,
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    )

    contrast = pd.DataFrame(np.mean(greycoprops(glcm, "contrast"), axis=1).tolist()).T
    contrast.columns = ["contrast_" + str(col) for col in lengths]
    
    dissimilarity = pd.DataFrame(
        np.mean(greycoprops(glcm, "dissimilarity"), axis=1).tolist()
    ).T
    dissimilarity.columns = ["dissimilarity_" + str(col) for col in lengths]
    
    homogeneity = pd.DataFrame(
        np.mean(greycoprops(glcm, "homogeneity"), axis=1).tolist()
    ).T
    homogeneity.columns = ["homogeneity_" + str(col) for col in lengths]
    
    ASM = pd.DataFrame(np.mean(greycoprops(glcm, "ASM"), axis=1).tolist()).T
    ASM.columns = ["asm_" + str(col) for col in lengths]
    
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


def peripherial_distribution_index(regionmask: np.ndarray, intensity: np.ndarray):
    """Computes peripherial distribution index of a grayscale image
    Ref PMID: 3116470
    """
    pass

def center_mismatch(regionmask: np.ndarray, intensity: np.ndarray):
    """ Compute distance between centroid and center of mass
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """
    regionmask=regionmask.astype('uint8')
    measures = measure.regionprops_table(regionmask,intensity, 
                         properties=['centroid','weighted_centroid'])
    dist = np.sqrt(np.square(measures['centroid-0']-measures['weighted_centroid-0'])+ np.square(measures['centroid-1']-measures['weighted_centroid-1']))[0]
    
    feat = {"center_mismatch": np.percentile(intensity[regionmask], 0)}
    return feat
    
    
def image_moments(regionmask: np.ndarray, intensity: np.ndarray):
    """ Compute image moments
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """
    
    moments_features = ['weighted_centroid','weighted_moments','weighted_moments_normalized',
                        'weighted_moments_central','weighted_moments_hu',
                        'moments','moments_normalized','moments_central','moments_hu']
    regionmask=regionmask.astype('uint8')

    feat = pd.DataFrame(measure.regionprops_table(regionmask,intensity,
                               properties=moments_features))

    return feat
    
    
    
def measure_texture_features(regionmask: np.ndarray, intensity: np.ndarray, lengths=[1, 5, 20]):
    """Compute all texture features
    This function computes all features that describe the image texture 
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales 
    Returns: A pandas dataframe with all the features for the given image
    """

    # compute features
    gclm_measures = gclm_textures(regionmask, intensity, lengths)
    
    moments_features = image_moments(regionmask, intensity)
    
    all_features = pd.concat([gclm_measures,moments_features], axis=1)
    
    return all_features