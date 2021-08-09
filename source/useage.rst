Example
=============

Below is a simple example demonstrating how to extract all available features. 

There are test images provided with this package.

.. code-block:: python

	#import libraries
	import sys
	sys.path.append("../")
	import os
	from nmco.utils.Run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext


	# initialising paths
	labelled_image_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/nuc_labels.tif')
	raw_image_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/raw_image.tif')
	feature_path = os.path.join(os.path.dirname(os.getcwd()),'example_data/')

	# For a quick extraction of all available features for all labelled nuclei given a segmented image with default parameters
	features = run_nuclear_chromatin_feat_ext(raw_image_path,labelled_image_path,feature_path)

Alternatively, use the CLI version using default parameters as follows. 

::
	python measure_nmco_features.py --rawdir <path/to/image> --datadir <path/to/labelled_image> --savedir <path/to/output/folder>



A more detailed implimenataion can be seen in the jupyter notebook "Nuclear_Features.ipynb"
