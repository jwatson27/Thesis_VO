import numpy as np
import os
import h5py
import keras
from keras.models import load_model
from src.helpers.helper_functions import undoNorm, loadNormParms, getBestValLoss, getFolder
from src.helpers.training_helpers import getGenerator
from src.helpers.custom_loss import scaledMSE_RT
from src.helpers.cfg import ThesisConfig

from src.helpers.helper_functions import getValLoss

# from keras.utils import plot_model
# from src.helpers.cfg import ThesisConfig
# test = 'CNN_test_12'
# configFile = os.path.join('exp_configs', test + '.yaml')
# config = ThesisConfig(configFile)
#
# name = config.expName
# numOutputs      = config.modelParms['numOutputs']
# defaultLossFunc = config.modelParms['defaultLossFunction']
# lossRotScale    = config.modelParms['lossRotScale']
# targetImageSize = config.thesisKittiParms['downsampledImageShape']
# numChannels   = config.modelParms['numImageChannels']
# useIMUData  = config.constraintParms['useIMU']
# useEpiRot   = config.constraintParms['useEpiRot']
# useEpiTrans = config.constraintParms['useEpiTrans']
#
# # Get Files
# checkpointFilesDict = config.trainPaths['checkpoint']
# figFilesDict = config.resultPaths['figures']
# figFolder = config.getFolderRef(config.getOutputFiles(figFilesDict, True))
#
#
# min_val_loss, min_val_epoch = getBestValLoss(config)
# print('Min Validation Loss: %s, Epoch %s' % (min_val_loss, min_val_epoch))
#
# # Loss Function
# if numOutputs>3:
#     lossFunc = scaledMSE_RT(lossRotScale)
#     keras.losses.lossFunction = lossFunc
# else:
#     lossFunc = defaultLossFunc
#
#
# # Model Epoch Selection
# checkpointFiles = config.getInputFiles(checkpointFilesDict)
# checkpointFolder = config.getFolderRef(checkpointFiles)
# checkpointFile = 'epoch_%03d%s' % (min_val_epoch, checkpointFilesDict['type'])
# modelFile = os.path.join(checkpointFolder, checkpointFile)
# model = load_model(modelFile)
# model.summary()
#
# plot_model(model, figFolder+'model.png', True)



def getPredictions(config, evalType='val', imu_bias_dph=None, imu_sensor_dpsh=None, saveData=False, batchSize=5, shuffle=False, epochNum=None):

    # Parameters
    name = config.expName
    numOutputs      = config.modelParms['numOutputs']
    defaultLossFunc = config.modelParms['defaultLossFunction']
    lossRotScale    = config.modelParms['lossRotScale']
    targetImageSize = config.thesisKittiParms['downsampledImageShape']
    numChannels   = config.modelParms['numImageChannels']
    useIMUData  = config.constraintParms['useIMU']
    useEpiRot   = config.constraintParms['useEpiRot']
    useEpiTrans = config.constraintParms['useEpiTrans']

    # Get Files
    checkpointFilesDict = config.trainPaths['checkpoint']
    if 'normParms' in config.kittiNormalized:
        normParmsFilesDict = config.kittiNormalized['normParms']
    else:
        normParmsFilesDict = config.trainPaths['normParms']
    imuFilesDict = config.kittiPrepared['imu']
    truthFilesDict = config.kittiPrepared['truth']
    epiFilesDict = config.kittiPrepared['epipolar']
    evalFilesDict = config.resultPaths['evaluations']
    normDataFilesDict  = config.kittiNormalized['normData']
    if 'normEpi' in config.kittiNormalized:
        normEpiFilesDict = config.kittiNormalized['normEpi']
    figFilesDict = config.resultPaths['figures']


    figFolder = config.getFolderRef(config.getOutputFiles(figFilesDict, True))

    if saveData:
        # Get Predictions Save File
        evalFolder = getFolder(evalFilesDict['dir'])

        predFilename = evalType + '_predictions'
        if imu_bias_dph is not None:
            predFilename += '_bias_'
            if not isinstance(imu_bias_dph, int) and not isinstance(imu_bias_dph, float):
                for bias in imu_bias_dph.ravel():
                    predFilename += ('%g' % bias).replace('.', 'p') + ','
                predFilename = predFilename.rstrip(',')
            else:
                predFilename += ('%g' % imu_bias_dph).replace('.', 'p')
        elif imu_sensor_dpsh is not None:
            predFilename += '_sensor_'
            if not isinstance(imu_sensor_dpsh, int) and not isinstance(imu_sensor_dpsh, float):
                for bias in imu_sensor_dpsh.ravel():
                    predFilename += ('%g' % bias).replace('.', 'p') + ','
                predFilename = predFilename.rstrip(',')
            else:
                predFilename += ('%g' % imu_sensor_dpsh).replace('.', 'p')
        if epochNum is not None:
            predFilename += '_epoch%03d' % epochNum;


        predictionsFile = os.path.join(evalFolder, predFilename + '.hdf5')

        if os.path.exists(predictionsFile):
            print('Predictions not saved. %s already exists' % predFilename)
            return


    if epochNum is None:
        val_loss_at_epoch, val_epoch = getBestValLoss(config)
        print('Min Validation Loss: %s, Epoch %s' % (val_loss_at_epoch, val_epoch))
        epochNum = val_epoch
    else:
        val_loss_at_epoch, val_epoch = getValLoss(config,epochNum)

    # Loss Function
    if numOutputs>3:
        lossFunc = scaledMSE_RT(lossRotScale)
        keras.losses.lossFunction = lossFunc
    else:
        lossFunc = defaultLossFunc


    # Model Epoch Selection
    checkpointFiles = config.getInputFiles(checkpointFilesDict)
    checkpointFolder = config.getFolderRef(checkpointFiles)
    checkpointFile = 'epoch_%03d%s' % (epochNum, checkpointFilesDict['type'])
    modelFile = os.path.join(checkpointFolder, checkpointFile)
    model = load_model(modelFile)


    # Get data generator
    evalGen = getGenerator(config, numOutputs, targetImageSize, numChannels, evalType,
                           batchSize=batchSize, imu_bias_error_dph=imu_bias_dph,
                           imu_sensor_error_dpsh=imu_sensor_dpsh, shuffleData=shuffle)



    # Calculate losses for all data points
    y_true_norm = np.empty((0,numOutputs))
    y_pred_norm = np.empty((0,numOutputs))
    total = len(evalGen)
    for idx, data in enumerate(evalGen):
        percentComplete = int(idx / total * 100)
        if divmod(idx, 300)[1] == 0:
            print('Percent Complete: %d%%' % percentComplete)

        x, y = data
        y_true_norm = np.append(y_true_norm, y, axis=0)
        y_pred_norm = np.append(y_pred_norm, model.predict(x), axis=0)



    # Undo normalization

    normParmsFile = config.getInputFiles(normParmsFilesDict)

    truth_rot_parms = loadNormParms(normParmsFile, 'rot_xyz')
    truth_xyz_parms = loadNormParms(normParmsFile, 'trans_xyz')
    truth_polar_parms = loadNormParms(normParmsFile, 'trans_rtp')

    if (numOutputs == 1):
        norm_parms = truth_polar_parms[:,0:1]
    elif (numOutputs == 3):
        norm_parms = truth_xyz_parms
    else:
        norm_parms = np.concatenate((truth_rot_parms,truth_xyz_parms), axis=1)

    y_true_real = undoNorm(y_true_norm, norm_parms)
    y_pred_real = undoNorm(y_pred_norm, norm_parms)


    if saveData:

        # Save Predictions for later
        with h5py.File(predictionsFile, 'w') as f:
            f.create_dataset('predictions', data=y_pred_real)
            f.create_dataset('truth', data=y_true_real)
            f.create_dataset('evalType', data=evalType)
            f.create_dataset('epoch', data=val_epoch)
            f.create_dataset('valLossAtEpoch', data=val_loss_at_epoch)
            if imu_bias_dph is not None:
                f.create_dataset('imu_bias_dph', data=imu_bias_dph)
            if imu_sensor_dpsh is not None:
                f.create_dataset('imu_sensor_dpsh', data=imu_sensor_dpsh)

        print('Saved predictions to %s' % predictionsFile)

    return (y_true_real, y_pred_real)



def loadPredFile(predFile):
    with h5py.File(predFile, 'r') as f:
        y_pred_real = np.array(f['predictions'])
        y_true_real = np.array(f['truth'])
        evalType = str(np.array(f['evalType']))
        min_val_epoch = np.array(f['epoch'])
        min_val_loss = np.array(f['valLossAtEpoch'])
    return (y_pred_real, y_true_real, evalType, min_val_epoch, min_val_loss)





def loadPredictions(tests, eType):
    errors_list = []
    preds_list = []
    truth_list = []
    filetruthrots_list = []


    for i, test in enumerate(tests):
        # if True:
        #     test = tests[0]
        configFile = 'exp_configs/%s.yaml' % test
        config = ThesisConfig(configFile)

        # Parameters
        name = config.expName
        numOutputs = config.modelParms['numOutputs']

        # Get Files
        evalFilesDict = config.resultPaths['evaluations']
        figFilesDict = config.resultPaths['figures']
        truthFilesDict = config.kittiPrepared['truth']

        figFolder = config.getFolderRef(config.getOutputFiles(figFilesDict, True))

        # Get Predictions Save File
        evalFolder = ''
        for pathSection in evalFilesDict['dir']:
            evalFolder = os.path.join(evalFolder, pathSection)
        predictionsFile = os.path.join(evalFolder, eType + '_predictions.hdf5')

        if os.path.exists(predictionsFile):

            # Load Predictions
            y_pred_real, y_true_real, evalType, min_val_epoch, min_val_loss = loadPredFile(predictionsFile)

            print('Min Validation Loss: %s, Epoch %s' % (min_val_loss, min_val_epoch))

            # get test idxs
            # numTested = y_true_real.shape[0]
            splitFilesDict = config.kittiPrepared['split']
            splitFile = config.getInputFiles(splitFilesDict)
            with h5py.File(splitFile, 'r') as f:
                if evalType == 'test':
                    turnIdxs = np.array(f['testTurnIdxs'])
                    nonTurnIdxs = np.array(f['testNonTurnIdxs'])
                elif evalType == 'val':
                    turnIdxs = np.array(f['valTurnIdxs'])
                    nonTurnIdxs = np.array(f['valNonTurnIdxs'])
            idxs = np.sort(np.concatenate((turnIdxs, nonTurnIdxs)))#[:numTested]

            truthData = np.empty((0, 3))
            for seq in config.usedSeqs:
                truthFile = config.getInputFiles(truthFilesDict, seq)
                with h5py.File(truthFile, 'r') as f:
                    rot_xyz = np.array(f['rot_xyz'])
                truthData = np.append(truthData, rot_xyz, axis=0)
            file_truth_rots = truthData[idxs, :]

            # Calculate average loss in each direction
            errors = y_true_real - y_pred_real

            errors_list.append(errors)
            preds_list.append(y_pred_real)
            truth_list.append(y_true_real)
            filetruthrots_list.append(file_truth_rots)

        else:
            print('predictions file %s does not exist' % predictionsFile)

    return(errors_list, preds_list, truth_list, filetruthrots_list)




