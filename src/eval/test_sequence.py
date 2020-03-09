import os
import numpy as np
import h5py
import keras
from keras.models import load_model
from src.helpers.cfg import ThesisConfig
from src.helpers.helper_functions import loadNormParms, applyNorm, undoNorm, getBestValLoss, getFolder
from src.helpers.custom_loss import scaledMSE_RT
from src.helpers.dataGenerator import DataGenerator


# Determine model to use for evaluation
# evalType = 'val'
evalType = 'test'




# tests = ['CNN_test_17', 'CNN_test_12']
# tests = ['trans_test_4', 'trans_test_2']
# tests = ['scale_test_4', 'scale_test_2']
tests = ['CNN_test_17', 'CNN_test_12', 'trans_test_4', 'trans_test_2', 'scale_test_4', 'scale_test_2']

for test in tests:

    # test = tests[0]
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)



    # LOAD PARAMETERS
    numOutputs = config.modelParms['numOutputs']
    useIMUData  = config.constraintParms['useIMU']
    targetImageSize = config.thesisKittiParms['downsampledImageShape']
    numChannels   = config.modelParms['numImageChannels']
    defaultLossFunc = config.modelParms['defaultLossFunction']
    lossRotScale    = config.modelParms['lossRotScale']




    # Read in sequence 10 input
    seq = 10
    cam = 0

    # FILE DICTIONARIES
    normImageFilesDict = config.kittiNormalized['normImages']
    truthFilesDict = config.kittiPrepared['truth']
    imuFilesDict = config.kittiPrepared['imu']
    normParmsFilesDict = config.kittiNormalized['normParms']
    checkpointFilesDict = config.trainPaths['checkpoint']


    # GET NORM PARAMETERS
    normParmsFile = config.getInputFiles(normParmsFilesDict)
    truth_rot_parms   = loadNormParms(normParmsFile, 'rot_xyz')
    truth_xyz_parms   = loadNormParms(normParmsFile, 'trans_xyz')
    truth_polar_parms = loadNormParms(normParmsFile, 'trans_rtp')
    imu_rot_parms     = loadNormParms(normParmsFile, 'noisy_rot_xyz')



    # GET IMAGES
    firstImageNames = np.empty(0)
    secondImageNames = np.empty(0)
    imageNames = config.getInputFiles(normImageFilesDict, seq, cam)
    firstImageNames = np.append(firstImageNames, imageNames[:-1], axis=0)
    secondImageNames = np.append(secondImageNames, imageNames[1:], axis=0)




    # GET TRUTH DATA
    truthData = np.empty((0,7))
    truthFile = config.getInputFiles(truthFilesDict, seq)
    with h5py.File(truthFile, 'r') as f:
        rot_xyz = np.array(f['rot_xyz'])
        trans_xyz = np.array(f['trans_xyz'])
        trans_rtp = np.array(f['trans_rtp'])

    # Normalize
    rot_xyz_norm = applyNorm(rot_xyz, truth_rot_parms)
    trans_xyz_norm = applyNorm(trans_xyz, truth_xyz_parms)
    trans_rtp_norm = applyNorm(trans_rtp, truth_polar_parms)

    trans_scale_norm = trans_rtp_norm[:, 0:1]
    rts = np.concatenate((rot_xyz_norm, trans_xyz_norm, trans_scale_norm), axis=1)
    truthData = np.append(truthData, rts, axis=0)

    if (numOutputs == 1):
        # get magnitude
        truthData = truthData[:, -1:]
    elif (numOutputs == 3):
        # get cartesian translation
        truthData = truthData[:, -4:-1]
    else:  # (numOut == 6)
        # get rotation and cartesian translation
        truthData = truthData[:, :-1]



    # GET IMU DATA
    imuData = None
    if useIMUData:
        imuData = np.empty((0, 3))
        imuFile = config.getInputFiles(imuFilesDict, seq)
        with h5py.File(imuFile, 'r') as f:
            noisy_rot_xyz = np.array(f['noisy_rot_xyz'])

        # Normalize
        noisy_rot_xyz_norm = applyNorm(noisy_rot_xyz, imu_rot_parms)
        imuData = np.append(imuData, noisy_rot_xyz_norm, axis=0)









    # Get predictions for sequence 10 test unshuffled


    seqGen = DataGenerator(configData=config,
                              turn_idxs=[0],
                              nonturn_idxs=np.array(range(1,len(firstImageNames))).tolist(),
                              prev_img_files=firstImageNames,
                              next_img_files=secondImageNames,
                              labels=truthData,
                              imu_xyz=imuData,
                              batch_size=1,
                              img_dim=targetImageSize,
                              n_channels=numChannels,
                              shuffle=False)


    # LOAD MODEL

    min_val_loss, min_val_epoch = getBestValLoss(config)
    print('Min Validation Loss: %s, Epoch %s' % (min_val_loss, min_val_epoch))

    # Loss Function
    if numOutputs > 3:
        lossFunc = scaledMSE_RT(lossRotScale)
        keras.losses.lossFunction = lossFunc
    else:
        lossFunc = defaultLossFunc


    checkpointFiles = config.getInputFiles(checkpointFilesDict)
    checkpointFolder = config.getFolderRef(checkpointFiles)
    checkpointFile = 'epoch_%03d%s' % (min_val_epoch, checkpointFilesDict['type'])
    modelFile = os.path.join(checkpointFolder, checkpointFile)
    model = load_model(modelFile)


    # PREDICT
    y_true_norm = np.empty((0, numOutputs))
    y_pred_norm = np.empty((0, numOutputs))
    total = len(seqGen)
    for idx, data in enumerate(seqGen):
        percentComplete = int(idx / total * 100)
        if divmod(idx, 50)[1] == 0:
            print('Percent Complete: %d%%' % percentComplete)

        x, y = data
        y_true_norm = np.append(y_true_norm, y, axis=0)
        y_pred_norm = np.append(y_pred_norm, model.predict(x), axis=0)




    # DENORMALIZE

    if (numOutputs == 1):
        truth_norm_parms = truth_polar_parms[:,0:1]
    elif (numOutputs == 3):
        truth_norm_parms = truth_xyz_parms
    else:
        truth_norm_parms = np.concatenate((truth_rot_parms,truth_xyz_parms), axis=1)

    y_true_real = undoNorm(y_true_norm, truth_norm_parms)
    y_pred_real = undoNorm(y_pred_norm, truth_norm_parms)


    # save everything to file

    evalFilesDict = config.resultPaths['evaluations']
    evalFolder = getFolder(evalFilesDict['dir'])
    predFilename = 'seq10_' + evalType + '_predictions'
    predictionsFile = os.path.join(evalFolder, predFilename + '.hdf5')
    if os.path.exists(predictionsFile):
        print('Predictions not saved. %s already exists' % predFilename)
    else:
        with h5py.File(predictionsFile, 'w') as f:
            f.create_dataset('predictions', data=y_pred_real)
            f.create_dataset('truth', data=y_true_real)
            f.create_dataset('evalType', data=evalType)
            f.create_dataset('epoch', data=min_val_epoch)
            f.create_dataset('valLossAtEpoch', data=min_val_loss)

        print('Saved predictions to %s' % predictionsFile)






