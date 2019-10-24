
# DONE:
# Configure code so that oversampling of turning/non-turning data can be turned on/off
# Reconfigure dataGenerator to work with turning/non-turning data
# Scripts to Normalize All Data
# Modify Generator to get normalized data


# TODO: Test Loss Function

# TODO: Make wrapper for run experiment to run multiple config files
# TODO: Make script to auto-generate multiple experiment config files for parameters sweeps


# TODO: Custom Loss Function
#   Just scaled rotation
#   With Constrain data in function
# TODO: Cost Function
#   -configure scale parameter so that the MSE values for rotation and translation are similar in size
#   -use constraint information for calculating errors in constraint calculations (i.e. epipolar errors compared to truth)


# TODO: Add functionality for using error truth labels
# TODO: Ensure that normalization occurs after truth labels have been calculated

# TODO: Get Epipolar R and t
# TODO: Modify Generator to work with IMU and Epipolar Data
# TODO: Modify Architecture function to account for IMU and Epipolar Data





# TODO: Research Quaternion Usage
# TODO: Research Architectures
# TODO: Architecture Configuraation Files/Scripts
# TODO: Main File
# TODO: Parameter Sweep Files

# TODO: Put code on github


# TODO: Reconfigure the following to work with turning/non-turning data
#   test_VO





# TODO: Get training working with config files
#   Create overall file that can run prep and training for a single experiment
#   Reconfigure master file to run multiple experiments


# TODO: Data preprocessing
#   Add functionality for removing rotation in the images for case 1


# TODO: Other ideas
#   Create config files for architecture models?
#   Create automatic plotting and diagram files

# TODO: Parameters to explore
#   Optimizer
#   Learning Rate / Momentum / Patience
#   CNN / Branch Architecture and Sizing
#   Dropout Amount / Placement
#   Cost Function Parameters (scaling on rotation)


# TODO: RESEARCH
#   -Is quaternion better than rot vec?
#   -Why do people use quaternion
#   -What range do people use to normalize image pixel values? (0 to 1) or (-1 to 1)


