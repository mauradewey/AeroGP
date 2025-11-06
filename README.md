## AeroGP

AeroGP is a climate emulator which predictes the global spatially resolved surface temperature response to regional anthropogenic aerosol perturbations. Please cite as:

Dewey, M., et al. (2025). AeroGP: Machine learning how aerosols impact regional climate. Journal of Geophysical Research: Machine Learning and Computation, 2, e2025JH000741. [doi.org/10.1029/2025JH000741](https://doi.org/10.1029/2025JH000741).

The processed training data and model code are also archived here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17099941.svg)](https://doi.org/10.5281/zenodo.17099941)

AeroGP can be run in either training or testing mode (in which case a trained version is loaded and run with some given input). In either case, the input data, output location, and the train/test toggle are all set in a config file. 

For example, the following line will train AeroGP with the regional 0x East Asian SO_2 experiment held out for validation: \
python AeroGP_SVGP_train_model.py ../config_files/main_config/ea0so2_config_lr.yml 
