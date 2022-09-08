[![Documentation Status](https://readthedocs.org/projects/nmco-image-features/badge/?version=latest)](https://nmco-image-features.readthedocs.io/en/latest/?badge=latest)

# chrometrics 
Since, the packing of the genome within the nucleus informs cellular state, nuclear morphology and chromatin organization(NMCO) features hold biologically meaningful information. High resolution images of DNA as visualised using a fluorescent microscope is a convenient tool to characterize such DNA organization. 

This package aims to provide an exhaustive set of interpretable morphometric and texture features for every single nucleus following segmentation from a 2D single channel image. Below is a brief overview of the features used 

<br/> 
<p align="center">
<img src='/NMCO_features.png' height='300' width='600'>
<br/>


Documentation is being updated and will be made available [here](https://nmco-image-features.readthedocs.io/en/latest/?badge=latest)

The list of features and their description can be found in the file "chrometric_feature_description.csv"

Illustration of feature extraction can also be checked here[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R9sddqwshbvrO6q04Jjd1QvANa9Jiy6M?authuser=1#scrollTo=tvQ3eDe69Ath)
   
## Installation 
The current implementation has been developed in Python 3.

In order to avoid any changes to the local packages, install in a virtual environment (optional).

```
   $ conda create --name nmco python
   $ conda activate nmco
```

To clone the repository run the following from the terminal.

```
   $ git clone https://github.com/GVS-Lab/chrometrics.git
```

Then install requirements and run the setup from the repository directory

```
   $ pip install -r requirements.txt
   $ python setup.py install
```

## Simple example 

```
#import libraries
import os
from nmco.utils.run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext


# initialising paths
labelled_image_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/nuc_labels.tif')
raw_image_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/raw_image.tif')
feature_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/')

# For a quick extraction of all available features for all labelled nuclei given a segmented image with default parameters
features = run_nuclear_chromatin_feat_ext(raw_image_path,labelled_image_path,feature_path)
```
Alternatively, use the CLI version using default parameters as follows. 

```
python measure_nmco_features.py --rawdir <path/to/image> --datadir <path/to/labelled_image> --savedir <path/to/output/folder>
```


## How to cite 

```bibtex
@article{venkatachalapathy2020multivariate,
  title={Multivariate analysis reveals activation-primed fibroblast geometric states in engineered 3D tumor microenvironments},
  author={Venkatachalapathy, Saradha and Jokhun, Doorgesh Sharma and Shivashankar, GV},
  journal={Molecular biology of the cell},
  volume={31},
  number={8},
  pages={803--812},
  year={2020},
  publisher={Am Soc Cell Biol}
}
```
