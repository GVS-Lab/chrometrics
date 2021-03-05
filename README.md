[![Documentation Status](https://readthedocs.org/projects/nmco-image-features/badge/?version=latest)](https://nmco-image-features.readthedocs.io/en/latest/?badge=latest)

# NMCO-Image-Features
The packing of the genome within the nucleus informs the cellular state. High resolution images of DNA as visualised using a fluorescent microscope is a convenient tool to characterize DNA organization. 

This package aims to provide an exhaustive set of interpretable morphometric and texture features for every single nucleus following segmentation from a 2D single channel image.

Documentation is available [here](https://nmco-image-features.readthedocs.io/en/latest/?badge=latest)

## Installation 
The current implementation has been developed in Python 3 and tested in Ubuntu 18.0


To clone the repository run the following from the terminal.

```
   $ git clone https://github.com/GVS-Lab/NMCO-Image-Features.git
```

Then run the setup from the repository directory

```
   $ sudo python setup.py install
```

In order to avoid any changes to the local packages, install in a virtual environment.

## Simple example 

```
#import libraries
import os
from nmco.utils.Run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext


# initialising paths
labelled_image_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/nuc_labels.tif')
raw_image_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/raw_image.tif')
feature_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/')

# For a quick extraction of all available features for all labelled nuclei given a segmented image with default parameters
features = run_nuclear_chromatin_feat_ext(raw_image_path,labelled_image_path,feature_path)
```
