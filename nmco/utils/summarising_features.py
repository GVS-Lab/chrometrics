# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def summarise_feature_table(data):
    """ Function that summarises distribution characteristics for all columns in a feature table. 
    Measures computed are median, min, max, standard deviation (SD) Coefficient of Variation (CV) and Coefficient of Dispersion (CD), Inter_Quartile_Range(IQR) and Quartile Coeeffient of Dispersrion (QCD).  
        
        Args: 
            data: feature table with the columns of interest plust label column for nuclear idenity
    """
    features = data.drop(["label"], axis=1).columns

    np.seterr(all="ignore")
    median_features = pd.DataFrame(
        np.array(np.nanmedian(data.drop(["label"], axis=1), axis=0))
    ).T
    median_features.columns = ["median_" + str(col) for col in features]

    min_features = pd.DataFrame(
        np.array(np.min(data.drop(["label"], axis=1), axis=0))
    ).T
    min_features.columns = ["min_" + str(col) for col in features]
    max_features = pd.DataFrame(
        np.array(np.max(data.drop(["label"], axis=1), axis=0))
    ).T
    max_features.columns = ["max_" + str(col) for col in features]

    SD_features = pd.DataFrame(np.array(np.std(data.drop(["label"], axis=1), axis=0))).T
    SD_features.columns = ["std_" + str(col) for col in features]
    CV_features = pd.DataFrame(
        np.array(np.std(data.drop(["label"], axis=1), axis=0))
        / np.array(np.nanmedian(data.drop(["label"], axis=1), axis=0))
    ).T
    CV_features.columns = ["CV_" + str(col) for col in features]
    CD_features = pd.DataFrame(
        np.array(np.var(data.drop(["label"], axis=1), axis=0))
        / np.array(np.nanmedian(data.drop(["label"], axis=1), axis=0))
    ).T
    CD_features.columns = ["CD_" + str(col) for col in features]
    IQR_features = pd.DataFrame(
        np.array(
            np.subtract(
                *np.nanpercentile(data.drop(["label"], axis=1), [75, 25], axis=0)
            )
        )
    ).T
    IQR_features.columns = ["IQR_" + str(col) for col in features]
    QCD_features = pd.DataFrame(
        np.array(
            np.subtract(
                *np.nanpercentile(data.drop(["label"], axis=1), [75, 25], axis=0)
            )
        )
        / np.array(
            np.add(*np.nanpercentile(data.drop(["label"], axis=1), [75, 25], axis=0))
        )
    ).T
    QCD_features.columns = ["QCD_" + str(col) for col in features]

    all_features = pd.concat(
        [
            median_features.reset_index(drop=True),
            min_features.reset_index(drop=True),
            max_features.reset_index(drop=True),
            SD_features.reset_index(drop=True),
            CV_features.reset_index(drop=True),
            CD_features.reset_index(drop=True),
            IQR_features.reset_index(drop=True),
            QCD_features.reset_index(drop=True),
        ],
        axis=1,
    )
    return all_features
