import os
import numpy as np
import glob
import h5py
import matplotlib.pyplot as plt

from src.helpers.cfg import ThesisConfig
from src.helpers.helper_functions import getFolder



# IMU tests: CNN_test_12, trans_test_2, scale_test_2

# Plot mean/std of errors vs bias values for each input axis and each output
#   (iR_x): oR_x, oR_y, oR_z, oT_x, oT_y, oT_z, etc...


tests = ['CNN_test_12', 'trans_test_2', 'scale_test_2']
evalType = 'test'
# evalType = 'val'
errorType = 'sensor'
# errorType = 'bias'

subfolder = 'test_arw'

for test in tests:
# test = tests[0]
# if True:
    print(test,evalType,errorType)
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    evalFilesDict = config.resultPaths['evaluations']

    evalFolder = getFolder(evalFilesDict['dir'])
    evalFolder = os.path.join(evalFolder,subfolder)

    numOutputs = config.modelParms['numOutputs']

    files = sorted(glob.glob(evalFolder + '/%s*%s_*%s' % (evalType, errorType, 'hdf5')))


    imu_error_vals = []
    error_rmses = []
    error_means = []
    error_stds = []
    for file in files:
        with h5py.File(file, 'r') as f:
            y_pred_real = np.array(f['predictions'])
            y_true_real = np.array(f['truth'])
            evalType = np.array(f['evalType'])
            min_val_epoch = np.array(f['epoch'])
            min_val_loss = np.array(f['valLossAtEpoch'])
            if 'imu_sensor_dpsh' in f:
                imu_error_amt = np.array([np.array(f['imu_sensor_dpsh'])])
            elif 'imu_bias_dph' in f:
                imu_error_amt = np.array([np.array(f['imu_bias_dph'])])

        imu_errors = []
        if len(imu_error_amt.shape)==1:
            imu_errors = imu_error_amt
        else:
            imu_errors = imu_error_amt[0,0,:]

        errors = y_true_real - y_pred_real
        imu_error_vals.append(imu_errors)
        error_rmses.append(np.sqrt(np.mean(errors**2, axis=0)))
        error_means.append(np.mean(errors, axis=0))
        error_stds.append(np.std(errors, axis=0))





    # Get test idxs where all three axes have the same amount of error
    mask = np.array([])
    for idx, vals in enumerate(imu_error_vals):
        mask = np.append(mask,len(vals))
        # if len(vals)==1:
        #     if vals[0]==0:
        #         allThreeZero_idx = idx


    # All Three Group
    allThree_idxs = np.argwhere(mask == 1).T[0, :]
    allThree_imu_error_vals = np.array(list(np.array(imu_error_vals)[allThree_idxs]))
    allThree_error_rmses = np.array(error_rmses)[allThree_idxs, :]
    allThree_error_means    = np.array(error_means)[allThree_idxs, :]
    allThree_error_stds     = np.array(error_stds)[allThree_idxs, :]

    sortOrder = np.argsort(allThree_imu_error_vals.T[0])
    allThree_imu_error_vals = allThree_imu_error_vals[sortOrder, :]
    allThree_error_rmses = allThree_error_rmses[sortOrder, :]
    allThree_error_means = allThree_error_means[sortOrder, :]
    allThree_error_stds = allThree_error_stds[sortOrder, :]

    allThree_saveFile = evalFolder + '/%s_predictions_%sError_three.%s' % (evalType, errorType, 'hdf5')
    with h5py.File(allThree_saveFile, 'w') as f:
        f.create_dataset('imu_error_vals', data=allThree_imu_error_vals)
        f.create_dataset('error_rmses', data=allThree_error_rmses)
        f.create_dataset('error_means', data=allThree_error_means)
        f.create_dataset('error_stds', data=allThree_error_stds)
    print('Saved data to %s' % allThree_saveFile)







    # Individual Groups
    individual_idxs = np.argwhere(mask != 1).T[0, :]
    individual_imu_error_vals = np.array(list(np.array(imu_error_vals)[individual_idxs]))
    individual_error_rmses    = np.array(error_rmses)[individual_idxs, :]
    individual_error_means    = np.array(error_means)[individual_idxs, :]
    individual_error_stds     = np.array(error_stds)[individual_idxs, :]

    for rowIdx in range(individual_imu_error_vals.shape[0]):
        compare = np.all((individual_imu_error_vals[rowIdx,:]==np.array([[0.,0.,0.]])))
        if compare:
            zerosIdx = rowIdx

    errorValsMask = (individual_imu_error_vals != 0)
    errorValsMask[zerosIdx,:] = np.array([True]*3)

    indiv_imu_error_vals = []
    indiv_error_rmses = []
    indiv_error_means = []
    indiv_error_stds = []
    for axis in range(3):
        axis_idxs = np.argwhere(errorValsMask[:, axis]).T[0, :]
        axis_imu_error_vals = individual_imu_error_vals[axis_idxs, axis:axis+1]
        axis_error_rmses = individual_error_rmses[axis_idxs, :]
        axis_error_means = individual_error_means[axis_idxs,:]
        axis_error_stds = individual_error_stds[axis_idxs,:]

        sortOrder = np.argsort(axis_imu_error_vals.T[0])
        axis_imu_error_vals = axis_imu_error_vals[sortOrder,0:1]
        axis_error_rmses = axis_error_rmses[sortOrder, :]
        axis_error_means = axis_error_means[sortOrder,:]
        axis_error_stds = axis_error_stds[sortOrder,:]

        indiv_imu_error_vals.append(axis_imu_error_vals)
        indiv_error_rmses.append(axis_error_rmses)
        indiv_error_means.append(axis_error_means)
        indiv_error_stds.append(axis_error_stds)

    indiv_imu_error_vals = np.array(indiv_imu_error_vals)
    indiv_error_rmses = np.array(indiv_error_rmses)
    indiv_error_means = np.array(indiv_error_means)
    indiv_error_stds = np.array(indiv_error_stds)



    individual_saveFile = evalFolder + '/%s_predictions_%sError_indiv.%s' % (evalType, errorType, 'hdf5')
    with h5py.File(individual_saveFile, 'w') as f:
        f.create_dataset('imu_error_vals', data=indiv_imu_error_vals)
        f.create_dataset('error_rmses', data=indiv_error_rmses)
        f.create_dataset('error_means', data=indiv_error_means)
        f.create_dataset('error_stds', data=indiv_error_stds)
    print('Saved data to %s' % individual_saveFile)

    # Save data to two different files






