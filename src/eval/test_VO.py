import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import keras
import numpy as np
import glob
import h5py
import os

from keras.utils import plot_model
from src.helpers.dataGenerator import DataGenerator
from src.prep.truthPoses_script import plotBirdsEye # TODO: Fix function
from src.helpers.coframes import cvtToAbs, sph2cart, getT_s_d, cvtToDcm_sd, getRT_sd_ss2d, cvtToRpy_sd
from src.helpers.helper_functions import load_model, load_history


# TODO: Update testing to work with new config data
# TODO: Update to work with config and turning/non-turning data

windowsRoot = '/windowsroot'
kittiPath = '%s/Pictures/Kitti Odometry/' % windowsRoot
imagePath = '%sdata_odometry_gray/%s/sequences/' % (kittiPath, 'downsampled')
indexFile = '%sdata_odometry_gray/normset/splitIndexes.hdf5' % (kittiPath)
posePath = '%sdata_odometry_poses/dataset/poses/' % (kittiPath)
imuPath = posePath



# Setup Information
numSeqs = 11
seqNames = ['%02d' % (seq) for seq in range(numSeqs)]
camNum = 0 # For a specific camera

target_image_size = (135, 480)
batchSize = 20
numChannels = 1
# imageShape = tuple(np.append(np.array(target_image_size),numChannels))
# numOutputs = 6
# useIMU = True
last_seq_length = 1590


print('Read in Image Names, Indexes, and Truth Data')

firstImageNames = np.empty(0)
secondImageNames = np.empty(0)
for seqStr in seqNames:
    # Get image size
    camPath = '%s%s/image_%s/' % (imagePath, seqStr, camNum)
    imageNames = sorted(glob.glob('%s*.png' % camPath))
    firstImageNames = np.concatenate((firstImageNames,imageNames[:-1]))
    secondImageNames = np.concatenate((secondImageNames,imageNames[1:]))

with h5py.File(indexFile, 'r') as f:
    # thesisTestIdxs = np.array(f['thesisTestIdxs'])
    projTrainIdxs  = np.array(f['projTrainIdxs'])
    projValIdxs    = np.array(f['projValIdxs'])
    projTestIdxs   = np.array(f['projTestIdxs'])

truthData = np.empty((0,6))
for seqStr in seqNames:
    truthFile = '%s%s-%s.hdf5' % (posePath, seqStr, camNum)
    with h5py.File(truthFile, 'r') as f:
        rottrans_xyzrtp = np.array(f['rottrans_xyzrtp'])
    truthData = np.concatenate((truthData,rottrans_xyzrtp),axis=0)

imuData = np.empty((0,3))
for seqStr in seqNames:
    imuFile = '%s%s-%s_rot_noise.hdf5' % (imuPath, seqStr, camNum)
    with h5py.File(imuFile, 'r') as f:
        noisy_rot_xyz = np.array(f['noisy_rot_xyz'])
    imuData = np.concatenate((imuData,noisy_rot_xyz),axis=0)



y_truth = truthData[projTestIdxs]


# Get test input
testGen_noimu = DataGenerator(input_idxs=projTestIdxs,
                         prev_img_files=firstImageNames,
                         next_img_files=secondImageNames,
                         labels=truthData,
                         batch_size=1,
                         dim=target_image_size,
                         imu_xyz=None,
                         shuffle=False,
                         n_channels=numChannels)
testGen_imu = DataGenerator(input_idxs=projTestIdxs,
                         prev_img_files=firstImageNames,
                         next_img_files=secondImageNames,
                         labels=truthData,
                         batch_size=1,
                         dim=target_image_size,
                         imu_xyz=imuData,
                         shuffle=False,
                         n_channels=numChannels)



normal_model_name = 'WatsonCNN_2_model_12'
normal_dir = os.path.join(os.getcwd(), 'saved_models', normal_model_name)
normal_model = load_model(save_dir=normal_dir, model_file_name='best_epoch.h5')
normal_cnn_model = normal_model.get_layer('CNN_Model')
# normal_preds = normal_model.predict_generator(testGen_noimu)
# np.save('normal_preds.npy', normal_preds)
plot_model(normal_model, show_shapes=True, to_file='%s/%s.png' % (normal_dir, normal_model_name))
plot_model(normal_cnn_model, show_shapes=True, to_file='%s/%s_cnn.png' % (normal_dir, normal_model_name))


imu_model_name = 'WatsonCNN_2_model_withIMU_1'
imu_dir = os.path.join(os.getcwd(), 'saved_models', imu_model_name)
imu_model = load_model(save_dir=imu_dir, model_file_name='best_epoch.h5')
imu_cnn_model = imu_model.get_layer('CNN_Model')
# imu_preds = imu_model.predict_generator(testGen_imu)
# np.save('imu_preds.npy', imu_preds)
plot_model(imu_model, show_shapes=True, to_file='%s/%s.png' % (imu_dir, imu_model_name))
plot_model(imu_cnn_model, show_shapes=True, to_file='%s/%s_cnn.png' % (imu_dir, imu_model_name))



# Get predictions for each model
# Normal Model
normal_preds = np.load('normal_preds.npy')
normal_mse = np.mean((y_truth-normal_preds)**2)

# IMU Model
imu_preds = np.load('imu_preds.npy')
imu_mse = np.mean((y_truth-imu_preds)**2)
#
# # TODO: Save Predictions to a File
#
#
# # Print MSE and RMSE Results for each model
print('Normal RMSE: %s meters' % (np.sqrt(normal_mse)))
print('IMU RMSE: %s meters' % (np.sqrt(imu_mse)))
#
#
# # Convert results back into camera frame
# # Convert Rotation rot_ip1_i
# normal_R_ip1_i = cvtToDcm_sd(normal_preds[:,:3])
# imu_R_ip1_i = cvtToDcm_sd(imu_preds[:,:3])
normal_R_ip1_i = cvtToDcm_sd(y_truth[:,:3])
imu_R_ip1_i = cvtToDcm_sd(y_truth[:,:3])
truth_R_ip1_i = cvtToDcm_sd(y_truth[:,:3])
#
# # Convert Translation rtp_ip1_ip12i
normal_t_ip1_ip12i = sph2cart(normal_preds[:,3:])
imu_t_ip1_ip12i = sph2cart(imu_preds[:,3:])
#
# # normal_t_ip1_ip12i = sph2cart(np.concatenate((normal_preds[:,3:4],y_truth[:,4:]), axis=1))
# # imu_t_ip1_ip12i = sph2cart(np.concatenate((imu_preds[:,3:4],y_truth[:,4:]), axis=1))
truth_t_ip1_ip12i = sph2cart(y_truth[:,3:])
#
# # Get T
normal_T_ip1_i = getT_s_d(normal_R_ip1_i, normal_t_ip1_ip12i)
imu_T_ip1_i = getT_s_d(imu_R_ip1_i, imu_t_ip1_ip12i)
truth_T_ip1_i = getT_s_d(truth_R_ip1_i, truth_t_ip1_ip12i)

normal_T_0_i = cvtToAbs(normal_T_ip1_i[-last_seq_length:])
imu_T_0_i = cvtToAbs(imu_T_ip1_i[-last_seq_length:])
truth_T_0_i = cvtToAbs(truth_T_ip1_i[-last_seq_length:])

print(len(normal_T_0_i))
print(len(imu_T_0_i))
print(len(truth_T_0_i))

truth_R_0_i, truth_t_0_02i = getRT_sd_ss2d(truth_T_0_i)
truth_rpy_0_i = cvtToRpy_sd(truth_R_0_i)
truth_yaw_i_0 = -truth_rpy_0_i[:,0,1]

normal_R_0_i, normal_t_0_02i = getRT_sd_ss2d(normal_T_0_i)
normal_rpy_0_i = cvtToRpy_sd(normal_R_0_i)
normal_yaw_i_0 = -normal_rpy_0_i[:,0,1]

imu_R_0_i, imu_t_0_02i = getRT_sd_ss2d(imu_T_0_i)
imu_rpy_0_i = cvtToRpy_sd(imu_R_0_i)
imu_yaw_i_0 = -imu_rpy_0_i[:,0,1]


# Plot birds eye view of results
plt.figure()
# plt.quiver(truth_t_0_02i[:,0], truth_t_0_02i[:,2], np.sin(np.unwrap(truth_yaw_i_0)), np.cos(np.unwrap(truth_yaw_i_0)))
# plt.quiver(normal_t_0_02i[:,0], normal_t_0_02i[:,2], np.sin(np.unwrap(normal_yaw_i_0)), np.cos(np.unwrap(normal_yaw_i_0)))
# plt.quiver(imu_t_0_02i[:,0], imu_t_0_02i[:,2], np.sin(np.unwrap(imu_yaw_i_0)), np.cos(np.unwrap(imu_yaw_i_0)))
# plt.legend(('Truth', 'Normal', 'IMU Aided'))
# plt.show()
# plt.savefig('test_birdseye.png')


plotBirdsEye(truth_T_0_i)
plotBirdsEye(normal_T_0_i)
plotBirdsEye(imu_T_0_i)
plt.legend(('Truth', 'Normal', 'IMU Aided'))
plt.xlabel('X Distance (m)')
plt.ylabel('Z Distance (m)')
plt.savefig('test_birdseye_just_trans.png')
plt.show()