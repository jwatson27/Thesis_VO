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




for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    evalFilesDict = config.resultPaths['evaluations']
    evalFolder = getFolder(evalFilesDict['dir'])

    numOutputs = config.modelParms['numOutputs']

    files = sorted(glob.glob(evalFolder + '/%s*%s*,*,*%s' % (evalType, 'sensor', 'hdf5')))

    axisDict = {'error': np.empty(0), 'means': np.empty((0,numOutputs)), 'stds': np.empty((0, numOutputs))}
    axisNames = ['x', 'y', 'z']
    dataDict = {}
    for axis in axisNames:
        dataDict[axis] = axisDict.copy()
    for file in files:
        with h5py.File(file, 'r') as f:
            y_pred_real = np.array(f['predictions'])
            y_true_real = np.array(f['truth'])
            evalType = np.array(f['evalType'])
            min_val_epoch = np.array(f['epoch'])
            min_val_loss = np.array(f['valLossAtEpoch'])
            imu_sensor_dpsh = np.array(f['imu_sensor_dpsh'])

        errors = y_true_real - y_pred_real

        error_means = np.mean(errors, axis=0)
        error_stds  = np.std(errors, axis=0)

        axisNums = np.nonzero(imu_sensor_dpsh)[1]
        if axisNums:
            axisNums = range(len(axisNames))

        for axisNum in axisNums:
            axis = axisNames[axisNum]
            dataDict[axis]['error']  = np.append(dataDict[axis]['error'], [imu_sensor_dpsh[0,axisNum]], axis=0)
            dataDict[axis]['means'] = np.append(dataDict[axis]['means'], [error_means], axis=0)
            dataDict[axis]['stds']  = np.append(dataDict[axis]['stds'], [error_stds], axis=0)

    # sort data
    for axis in dataDict:
        sortOrder = np.argsort(dataDict[axis]['error'])
        for key in dataDict[axis]:
            dataDict[axis][key] = dataDict[axis][key][sortOrder]


    # plot data
    errorMinMax = [0, 3000]
    colors = ['r', 'g', 'b']
    styleList = ['-', '--', ':', '-', '--', ':']
    markerList = ['None', 'None', 'None', 'p', 'p', 'p']
    style = styleList[:numOutputs]
    marker = markerList[:numOutputs]
    labels = ['rotX', 'rotY', 'rotZ', 'transX', 'transY', 'transZ']
    if numOutputs==3:
        labels = labels[3:]
    elif numOutputs==1:
        labels = ['scale']

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for axisNum, axis in enumerate(axisNames):
        error = dataDict[axis]['error']
        means = dataDict[axis]['means']
        stds = dataDict[axis]['stds']

        minIdxs = np.argwhere(error >= errorMinMax[0])
        maxIdxs = np.argwhere(error <= errorMinMax[1])
        firstIdx = 0
        if len(minIdxs)>0:
            firstIdx = minIdxs[0,0]
        lastIdx = len(error)
        if len(maxIdxs)>0:
            lastIdx = maxIdxs[-1,0]

        for outputNum in range(numOutputs):
            # mean plot
            ax1.semilogx(error[firstIdx:lastIdx+1], means[firstIdx:lastIdx+1,outputNum],
                         colors[axisNum]+style[outputNum], label=axis+'-'+labels[outputNum], marker=marker[outputNum])

            # std dev plot
            ax2.semilogx(error[firstIdx:lastIdx+1], stds[firstIdx:lastIdx+1,outputNum],
                         colors[axisNum]+style[outputNum], label=axis+'-'+labels[outputNum], marker=marker[outputNum])

    # # plot negative data
    errorMinMax = [-3000, 0]
    colors = ['k', 'k', 'k']

    for axisNum, axis in enumerate(axisNames):
        error = dataDict[axis]['error']
        means = dataDict[axis]['means']
        stds = dataDict[axis]['stds']

        minIdxs = np.argwhere(error >= errorMinMax[0])
        maxIdxs = np.argwhere(error <= errorMinMax[1])
        firstIdx = 0
        if len(minIdxs) > 0:
            firstIdx = minIdxs[0, 0]
        lastIdx = len(error)
        if len(maxIdxs) > 0:
            lastIdx = maxIdxs[-1, 0]

        for outputNum in range(numOutputs):
            # mean plot
            ax1.semilogx(-error[firstIdx:lastIdx + 1], means[firstIdx:lastIdx + 1, outputNum],
                         colors[axisNum] + style[outputNum], label=axis + '-' + labels[outputNum],
                         marker=marker[outputNum])

            # std dev plot
            ax2.semilogx(-error[firstIdx:lastIdx + 1], stds[firstIdx:lastIdx + 1, outputNum],
                         colors[axisNum] + style[outputNum], label=axis + '-' + labels[outputNum],
                         marker=marker[outputNum])


    ax1.set_ylabel('Error Mean')
    # ax1.legend()
    ax1.set_title('%s - Mean and Std Dev of Errors vs. Sensor Error' % test)

    ax2.set_xlabel('Sensor Error (deg/$\sqrt{hr}$)')
    ax2.set_ylabel('Error Std Dev')
    ax2.legend()

plt.show(block=True)