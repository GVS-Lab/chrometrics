import argparse
from nmco.utils.run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext
                             
# Parse the input arguments
options = argparse.ArgumentParser(description = "Extract NMCO features from segmented nuclei")

options.add_argument('--rawdir', help = 'directory of DNA images' , default = 'example_data/raw_image.tif')
options.add_argument('--datadir', help = 'directory of segmented images' , default = 'example_data/nuc_labels.tif')
options.add_argument('--savedir', help = 'directory to store feature tables', default = 'example_data/')

options.add_argument('--calliper_angular_resolution', type = int, help = 'Angular resolution for obtaining caliper distances',
                     default = 10)
options.add_argument('--measure_simple_geometry', type = bool, help = 'Measure simple geometric features', default = True)
options.add_argument('--measure_calliper_distances', type = bool, help = 'Measure caliper distances', default = True)
options.add_argument('--measure_radii_features', type = bool, help = 'Measure radii distances', default = True)

options.add_argument('--step_size_curvature', type = int, help = 'Step size for computing local curvature', default = 2)
options.add_argument('--prominance_curvature', type = float, help = 'Prominance for detecting peaks in curvature', 
                     default = 0.1)
options.add_argument('--width_prominent_curvature', type = int, help = 'Minimum width for detecting peaks in curvature', 
                     default = 5)
options.add_argument('--dist_bt_peaks_curvature', type = int, help = 'Minimum distance between peaks', 
                     default = 10)

options.add_argument('--measure_int_dist_features', type = bool, help = 'Measure intensity distribution features', 
                     default = True)
options.add_argument('--measure_hc_ec_ratios_features', type = bool, help = 'Measure Heterochromatin to Euchromatin ratios', 
                     default = True)
options.add_argument('--hc_threshold', type = float, help = 'threshold (alpha) to detect heterochromatin', 
                     default = 1)

options.add_argument('--gclm_lengths', type = int, nargs='+', help = 'list of lengths to be used for GCLM features', 
                     default = [1, 5, 20])
options.add_argument('--measure_gclm_features', type = bool, help = 'Measure GCLM features', 
                     default = True)
options.add_argument('--measure_moments_features', type = bool, help = 'Measure image moments', 
                     default = True)


arguments = options.parse_args()

# compute features for an image
run_nuclear_chromatin_feat_ext(raw_image_path = arguments.rawdir,
                               labelled_image_path = arguments.datadir,
                               output_dir = arguments.savedir,
                               calliper_angular_resolution = arguments.calliper_angular_resolution,
                               measure_simple_geometry = arguments.measure_simple_geometry,
                               measure_calliper_distances = arguments.measure_calliper_distances,
                               measure_radii_features = arguments.measure_radii_features,
                               step_size_curvature = arguments.step_size_curvature,
                               prominance_curvature = arguments.prominance_curvature,
                               width_prominent_curvature = arguments.width_prominent_curvature, 
                               dist_bt_peaks_curvature = arguments.dist_bt_peaks_curvature, 
                               measure_int_dist_features = arguments.measure_int_dist_features, 
                               measure_hc_ec_ratios_features = arguments.measure_hc_ec_ratios_features, 
                               hc_threshold = arguments.hc_threshold, 
                               gclm_lengths = arguments.gclm_lengths,
                               measure_gclm_features = arguments.measure_gclm_features,
                               measure_moments_features = arguments.measure_moments_features
                              )
                                
