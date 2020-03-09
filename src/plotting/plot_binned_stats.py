from src.helpers.cfg import ThesisConfig
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from src.helpers.plot_helpers import plotHistAndGuassian, plotTrueVsPredAndCounts, getTitleTypesAndUnits
from src.helpers.helper_functions import getFolder



# Determine model to use for evaluation
test = 'scale_test_2'
evalType = 'val'

savePlots = False
blocking = True

# compareData='predictions'
compareData='epipolar'








# GET DATA
y_pred_real = None
y_true_real = None
if compareData=='predictions':
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    # Parameters
    numOutputs = config.modelParms['numOutputs']

    # GET TRUTH AND PREDICTIONS
    evalFilesDict = config.resultPaths['evaluations']
    evalFolder = getFolder(evalFilesDict['dir'])
    predictionsFile = os.path.join(evalFolder, evalType + '_predictions.hdf5')
    if os.path.exists(predictionsFile):
        # Load Predictions
        with h5py.File(predictionsFile, 'r') as f:
            y_pred_real = np.array(f['predictions'])
            y_true_real = np.array(f['truth'])
            evalType = np.array(f['evalType'])
            min_val_epoch = np.array(f['epoch'])
            min_val_loss = np.array(f['valLossAtEpoch'])
        print('Min Validation Loss: %s, Epoch %s' % (min_val_loss, min_val_epoch))
    cmpTitle='True vs. Predicted'

elif compareData=='epipolar':
    test = 'CNN_test_13'
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    # Parameters
    numOutputs = config.modelParms['numOutputs']

    # GET TRUTH DATA
    truthFilesDict = config.kittiPrepared['truth']

    true_xyz = np.empty((0, 6))
    for cam in config.usedCams:
        for seq in config.usedSeqs:
            truthFile = config.getInputFiles(truthFilesDict, seq)
            with h5py.File(truthFile, 'r') as f:
                true_rot_xyz = np.array(f['rot_xyz'])
                true_trans_xyz = np.array(f['trans_xyz'])
            true_trans_xyz_norm = true_trans_xyz / np.array([np.sqrt(np.sum(true_trans_xyz ** 2, axis=1))]).T
            true = np.concatenate((true_rot_xyz, true_trans_xyz_norm), axis=1)
            true_xyz = np.append(true_xyz, true, axis=0)

    y_true_real = true_xyz

    # GET EPIPOLAR DATA
    epiFilesDict = config.kittiPrepared['epipolar']

    epi_xyz = np.empty((0, 6))
    epi_xyz_masked = np.empty((0, 6))
    epi_xyz_masked1000 = np.empty((0, 6))
    epi_xyz_1000 = np.empty((0, 6))
    for cam in config.usedCams:
        for seq in config.usedSeqs:
            epiFile = config.getInputFiles(epiFilesDict, seq, cam)
            with h5py.File(epiFile, 'r') as f:
                epi_rot_xyz = np.array(f['epi_rot_xyz'])
                epi_trans_xyz = np.array(f['epi_trans_xyz'])
            epi = np.concatenate((epi_rot_xyz, epi_trans_xyz),axis=1)
            epi_xyz = np.append(epi_xyz, epi, axis=0)

            epiFileMasked = '.'.join((epiFile.rsplit('.')[0] + '_masked', epiFile.rsplit('.')[1]))
            with h5py.File(epiFileMasked, 'r') as f:
                epi_rot_xyz_masked = np.array(f['epi_rot_xyz'])
                epi_trans_xyz_masked = np.array(f['epi_trans_xyz'])
            epi_masked = np.concatenate((epi_rot_xyz_masked, epi_trans_xyz_masked),axis=1)
            epi_xyz_masked = np.append(epi_xyz_masked, epi_masked, axis=0)

            # epiFileMasked1000 = '.'.join((epiFile.rsplit('.')[0] + '_masked1000', epiFile.rsplit('.')[1]))
            # with h5py.File(epiFileMasked1000, 'r') as f:
            #     epi_rot_xyz_masked1000 = np.array(f['epi_rot_xyz'])
            #     epi_trans_xyz_masked1000 = np.array(f['epi_trans_xyz'])
            # epi_masked1000 = np.concatenate((epi_rot_xyz_masked1000, epi_trans_xyz_masked1000), axis=1)
            # epi_xyz_masked1000 = np.append(epi_xyz_masked1000, epi_masked1000, axis=0)

            # epiFile1000 = '.'.join((epiFile.rsplit('.')[0] + '_1000', epiFile.rsplit('.')[1]))
            # with h5py.File(epiFile1000, 'r') as f:
            #     epi_rot_xyz_1000 = np.array(f['epi_rot_xyz'])
            #     epi_trans_xyz_1000 = np.array(f['epi_trans_xyz'])
            # epi_1000 = np.concatenate((epi_rot_xyz_1000, epi_trans_xyz_1000), axis=1)
            # epi_xyz_1000 = np.append(epi_xyz_1000, epi_1000, axis=0)

    y_pred_real = epi_xyz  # not masked
    y_pred_real = epi_xyz_masked  # masked

    numBins = 100
    minNumPerBin = 10
    cmpTitle = 'Masked vs. Epipolar'



if savePlots:
    # Figure Save Location
    figFilesDict = config.resultPaths['figures']
    figFolder = getFolder(figFilesDict['dir'])



# TODO: STOPPING SPOT, trying to figure out if epipolar data calculated with fewer features (i.e. 1000) helps to remove ouliers in the masked epipolar values
# i = 2
# plotTrueVsPredAndCounts(y_true_real[:,i], epi_xyz[:,i], numBins, minNumPerBin)
# plotTrueVsPredAndCounts(y_true_real[:,i], epi_xyz_masked[:,i], numBins, minNumPerBin)
# plotTrueVsPredAndCounts(y_true_real[:,i], epi_xyz_masked1000[:,i], numBins, minNumPerBin)
# plotTrueVsPredAndCounts(y_true_real[:,i], epi_xyz_1000[:,i], numBins, minNumPerBin)
# plt.show(block=blocking)


errors = y_true_real - y_pred_real
threshs = [0.05, 0.05, 0.05]

outlier_idx_lists = []
for i in range(3):
    outlier_idxs = np.argwhere(abs(errors[:,i]) > threshs[i])[:,0]
    outlier_idx_lists.append(outlier_idxs)

total_outlier_idxs = np.empty((0))
for lst in outlier_idx_lists:
    total_outlier_idxs = np.append(total_outlier_idxs, lst)
total_outlier_idxs = np.array(list(set(total_outlier_idxs)))

filt_true = []
filt_epi = []
for i in range(3):
    filtered_true = np.delete(y_true_real[:,i], total_outlier_idxs)
    filtered_epi = np.delete(y_pred_real[:,i], total_outlier_idxs)
    filt_true.append(filtered_true)
    filt_epi.append(filtered_epi)


from src.helpers.kitti_helpers import IndexConverter
truthFilesDict = config.kittiPrepared['truth']
truthFilesList = []
for seq in config.usedSeqs:
    truthFilesList.append(config.getInputFiles(truthFilesDict, seq))
idxCvt = IndexConverter(truthFilesList)

absIdxs = idxCvt.cvtToSeqs(outlier_idx_lists[2])
absolute = {}
for i in range(len(absIdxs)):
    absolute[str(i)] = absIdxs[i]

# Seq 01, pair 1098  |  ON HIGHWAY, NO APPARENT REASON FOR FAILURE
# Seq 02, pair 1793  |  FOLLOWING MOTORCYCLE, NO APPARENT REASON FOR FAILURE
# Seq 08, pair 4013  |  IN THE MIDDLE OF STOPPING, WHY DID OUTLIER NOT OCCUR ON OTHER FRAMES?



# rotX: (08, 4013)
# rotY: (01, 1098), (02, 1793), (08, 4013)
# rotZ: (08, 4013)


# if True:
#     i = 0
for i in range(3):
    plotTrueVsPredAndCounts(y_true_real[:,i], y_pred_real[:,i], numBins, minNumPerBin)
    ax1 = plt.gcf().axes[0]
    ax1.plot(y_true_real[outlier_idx_lists[i],i], y_pred_real[outlier_idx_lists[i],i], 'ro')

    # plotTrueVsPredAndCounts(filt_true[i], filt_epi[i], numBins, minNumPerBin)
    # # ax1 = plt.gcf().axes[0]
    # # ax1.plot(y_true_real[outlier_idx_lists[i],i], y_pred_real[outlier_idx_lists[i],i], 'ro')
plt.show(block=blocking)


y_true_real = np.array(filt_true).T
y_pred_real = np.array(filt_epi).T


if y_true_real is not None and y_pred_real is not None:
    # PLOT RESULTS
    title_types, units = getTitleTypesAndUnits(numOutputs)

    # True vs. Compare Plot
    for i in range(y_true_real.shape[1]):
        saveCountsFile = None
        if savePlots:
            saveCountsFile = os.path.join(figFolder, title_types[i] + '.png')

        plotTrueVsPredAndCounts(y_true_real[:,i], y_pred_real[:,i], numBins, minNumPerBin, name=config.expName,
                                title=cmpTitle, title_type=title_types[i], units=units[i],
                                saveFile=saveCountsFile)
    plt.show(block=blocking)

    # Error Plot
    errors = y_true_real - y_pred_real
    for i in range(errors.shape[1]):
        saveErrorFile = None
        if savePlots:
            saveErrorFile = os.path.join(figFolder, title_types[i] + '_errors.png')

        plotHistAndGuassian(errors[:,i], numBins, name=config.expName, title='Error Histogram',
                            title_type=title_types[i], units=units[i], saveFile=saveErrorFile)
    plt.show(block=blocking)





















