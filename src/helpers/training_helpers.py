from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.helpers.custom_callbacks import SaveHistoryToFile, PlotHistory
import os
import h5py
import numpy as np
from src.helpers.dataGenerator import DataGenerator
from src.helpers.helper_functions import undoNorm, applyNorm, loadNormParms


def getCallbacksList(config, historyFilepath, checkpointFilepath):

    # Parameters

    # Callbacks
    # Model Checkpoint
    use_checkpoint_clbk  = config.checkpointParms['useCallback']
    checkpoint_verbosity = config.checkpointParms['verbosity']
    # Early Stopping
    use_earlyStopping_clbk  = config.earlyStoppingParms['useCallback']
    earlyStopping_monitor   = config.earlyStoppingParms['monitor']
    earlyStopping_minChange = config.earlyStoppingParms['minChange']
    earlyStopping_patience  = config.earlyStoppingParms['patience']
    earlyStopping_verbosity = config.earlyStoppingParms['verbosity']
    earlyStopping_baseline  = config.earlyStoppingParms['baseline']
    # Reduce Learning Rate
    use_reduceLR_clbk  = config.reduceLRParms['useCallback']
    reduceLR_monitor   = config.reduceLRParms['monitor']
    reduceLR_factor    = config.reduceLRParms['factor']
    reduceLR_patience  = config.reduceLRParms['patience']
    reduceLR_verbosity = config.reduceLRParms['verbosity']
    reduceLR_minChange = config.reduceLRParms['minChange']
    reduceLR_cooldown  = config.reduceLRParms['cooldown']
    reduceLR_minLR     = config.reduceLRParms['minLR']
    # Plot History
    use_plotHistory_clbk    = config.plotHistoryParms['useCallback']
    plotHistory_lossFig     = config.plotHistoryParms['plotLossFigure']
    plotHistory_monitorList = config.plotHistoryParms['monitorList']
    plotHistory_windowSize  = config.plotHistoryParms['windowSize']
    plotHistory_title       = config.plotHistoryParms['title']
    plotHistory_xlabel      = config.plotHistoryParms['xlabel']
    plotHistory_ylabel      = config.plotHistoryParms['ylabel']




    callbacksList = []

    # CUSTOM: SAVE HISTORY TO FILE
    history_clbk = SaveHistoryToFile(historyFilepath)
    callbacksList.append(history_clbk)

    # MODEL CHECKPOINT
    if (use_checkpoint_clbk):
        checkpoint_clbk = ModelCheckpoint(filepath=checkpointFilepath,
                                          verbose=checkpoint_verbosity)
        callbacksList.append(checkpoint_clbk)

    # EARLY STOPPING
    if (use_earlyStopping_clbk):
        earlyStopping_clbk = EarlyStopping(monitor=earlyStopping_monitor,
                                           min_delta=earlyStopping_minChange,
                                           patience=earlyStopping_patience,
                                           verbose=earlyStopping_verbosity,
                                           baseline=earlyStopping_baseline)
        callbacksList.append(earlyStopping_clbk)

    # REDUCE LR ON PLATEU
    if (use_reduceLR_clbk):
        reduceLR_clbk = ReduceLROnPlateau(monitor=reduceLR_monitor,
                                          factor=reduceLR_factor,
                                          patience=reduceLR_patience,
                                          verbose=reduceLR_verbosity,
                                          min_delta=reduceLR_minChange,
                                          cooldown=reduceLR_cooldown,
                                          min_lr=reduceLR_minLR)
        callbacksList.append(reduceLR_clbk)

    # CUSTOM: PLOT HISTORY
    if (use_plotHistory_clbk):
        plotHistory_clbk = PlotHistory(filepath=historyFilepath,
                                       plotLossFigure=plotHistory_lossFig,
                                       monitorList=plotHistory_monitorList,
                                       windowSize=plotHistory_windowSize,
                                       title=plotHistory_title,
                                       xlabel=plotHistory_xlabel,
                                       ylabel=plotHistory_ylabel)
        callbacksList.append(plotHistory_clbk)


    return callbacksList




def getTrainAndValGenerators(config, numOut, targetImgSize, numChan):

    # Parameters
    # dataset
    usedCams    = config.usedCams
    usedSeqs    = config.usedSeqs

    # training
    batchSize   = config.trainingParms['batchSize']
    valBatchSize = config.trainingParms['valBatchSize']
    oversampTurnFrac = config.trainingParms['oversampTurnFrac']

    # constraints
    useIMUData  = config.constraintParms['useIMU']
    useEpiRot   = config.constraintParms['useEpiRot']
    useEpiTrans = config.constraintParms['useEpiTrans']

    # normalization
    useNormImages = config.normalizationParms['useNormImages']
    useNormTruth  = config.normalizationParms['useNormTruth']
    useNormIMU    = config.normalizationParms['useNormIMU']
    useNormEpi    = config.normalizationParms['useNormEpi']


    # Files
    truthFilesDict = config.kittiPrepared['truth']
    imuFilesDict = config.kittiPrepared['imu']
    epiFilesDict = config.kittiPrepared['epipolar']
    normImageFilesDict = config.kittiNormalized['normImages']
    normDataFilesDict  = config.kittiNormalized['normData']
    if 'normEpi' in config.kittiNormalized:
        normEpiFilesDict = config.kittiNormalized['normEpi']


    # Split Indexes
    splitFilesDict = config.kittiPrepared['split']
    splitFile = config.getInputFiles(splitFilesDict)
    with h5py.File(splitFile, 'r') as f:
        trainTurnIdxs = np.array(f['trainTurnIdxs'])
        valTurnIdxs = np.array(f['valTurnIdxs'])
        # testTurnIdxs = np.array(f['testTurnIdxs'])
        trainNonTurnIdxs = np.array(f['trainNonTurnIdxs'])
        valNonTurnIdxs = np.array(f['valNonTurnIdxs'])
        # testNonTurnIdxs = np.array(f['testNonTurnIdxs'])


    if useNormImages:
        # Image Files - Normalized
        firstImageNames = np.empty(0)
        secondImageNames = np.empty(0)
        for cam in usedCams:
            for seq in usedSeqs:
                imageNames = config.getInputFiles(normImageFilesDict, seq, cam)
                firstImageNames = np.append(firstImageNames, imageNames[:-1], axis=0)
                secondImageNames = np.append(secondImageNames, imageNames[1:], axis=0)

    # Get Truth Data
    truthData = np.empty((0,7))
    if useNormTruth:
        # Normalized
        normDataFile = config.getInputFiles(normDataFilesDict)
        with h5py.File(normDataFile, 'r') as f:
            norm_rot_xyz = np.array(f['rot_xyz'])
            norm_trans_xyz = np.array(f['trans_xyz'])
            norm_trans_rtp = np.array(f['trans_rtp'])
        norm_trans_scale = norm_trans_rtp[:, 0:1]
        norm_rts = np.concatenate((norm_rot_xyz, norm_trans_xyz, norm_trans_scale), axis=1)
        truthData = np.append(truthData, norm_rts, axis=0)
    else:
        # Original
        for seq in usedSeqs:
            truthFile = config.getInputFiles(truthFilesDict, seq)
            with h5py.File(truthFile, 'r') as f:
                rot_xyz = np.array(f['rot_xyz'])
                trans_xyz = np.array(f['trans_xyz'])
                trans_rtp = np.array(f['trans_rtp'])
            trans_scale = trans_rtp[:, 0:1]
            rts = np.concatenate((rot_xyz, trans_xyz, trans_scale), axis=1)
            truthData = np.append(truthData, rts, axis=0)

    if (numOut == 1):
        # get magnitude
        truthData = truthData[:, -1:]
    elif (numOut == 3):
        # get cartesian translation
        truthData = truthData[:, -4:-1]
    else:  # (numOut == 6)
        # get rotation and cartesian translation
        truthData = truthData[:, :-1]


    imuData = None
    if useIMUData:
        # Get IMU Data
        if useNormIMU:
            normDataFile = config.getInputFiles(normDataFilesDict)
            # Normalized
            imuData = np.empty((0, 3))
            with h5py.File(normDataFile, 'r') as f:
                norm_imu_rot = np.array(f['noisy_rot_xyz'])
            imuData = np.append(imuData, norm_imu_rot, axis=0)
        else:
            # Original
            imuData = np.empty((0, 3))
            for seq in usedSeqs:
                imuFile = config.getInputFiles(imuFilesDict, seq)
                with h5py.File(imuFile, 'r') as f:
                    noisy_rot_xyz = np.array(f['noisy_rot_xyz'])
                imuData = np.append(imuData, noisy_rot_xyz, axis=0)


    epiRotData = None
    if useEpiRot:
        # Get Epipolar Rotation Data
        epiRotData = np.empty((0,3))
        if useNormEpi:
            # Normalized
            normEpiFile = config.getInputFiles(normEpiFilesDict)
            with h5py.File(normEpiFile, 'r') as f:
                norm_epi_rot = np.array(f['epi_rot_xyz'])
            epiRotData = np.append(epiRotData, norm_epi_rot, axis=0)
        else:
            # Original
            for cam in usedCams:
                for seq in usedSeqs:
                    epiFile = config.getInputFiles(epiFilesDict, seq, cam)
                    with h5py.File(epiFile, 'r') as f:
                        epi_rot_xyz = np.array(f['epi_rot_xyz'])
                    epiRotData = np.append(epiRotData, epi_rot_xyz, axis=0)


    epiTransData = None
    if useEpiTrans:
        # Get Epipolar Translation Data
        epiTransData = np.empty((0,3))
        if useNormEpi:
            # Normalized
            normEpiFile = config.getInputFiles(normEpiFilesDict)
            with h5py.File(normEpiFile, 'r') as f:
                norm_epi_trans = np.array(f['epi_trans_xyz'])
            epiTransData = np.append(epiTransData, norm_epi_trans, axis=0)
        else:
            # Original
            for cam in usedCams:
                for seq in usedSeqs:
                    epiFile = config.getInputFiles(epiFilesDict, seq, cam)
                    with h5py.File(epiFile, 'r') as f:
                        epi_trans_xyz = np.array(f['epi_trans_xyz'])
                    epiTransData = np.append(epiTransData, epi_trans_xyz, axis=0)




    trainGenerator = DataGenerator(configData=config,
                                    turn_idxs=trainTurnIdxs,
                                    nonturn_idxs=trainNonTurnIdxs,
                                    prev_img_files=firstImageNames,
                                    next_img_files=secondImageNames,
                                    labels=truthData,
                                    frac_turn=oversampTurnFrac,
                                    imu_xyz=imuData,
                                    epi_rot=epiRotData,
                                    epi_trans=epiTransData,
                                    batch_size=batchSize,
                                    img_dim=targetImgSize,
                                    n_channels=numChan)

    valGenerator = DataGenerator(configData=config,
                                    turn_idxs=valTurnIdxs,
                                    nonturn_idxs=valNonTurnIdxs,
                                    prev_img_files=firstImageNames,
                                    next_img_files=secondImageNames,
                                    labels=truthData,
                                    frac_turn=None,
                                    imu_xyz=imuData,
                                    epi_rot=epiRotData,
                                    epi_trans=epiTransData,
                                    batch_size=valBatchSize,
                                    img_dim=targetImgSize,
                                    n_channels=numChan)

    return (trainGenerator, valGenerator)




def getGenerator(config, numOut, targetImgSize, numChan, genType='train', batchSize=None, shuffleData=True, imu_bias_error_dph=None, imu_sensor_error_dpsh=None):

    # Parameters
    # dataset
    usedCams    = config.usedCams
    usedSeqs    = config.usedSeqs

    # training
    if batchSize is None:
        if genType == 'train':
            batchSize   = config.trainingParms['batchSize']
        if genType == 'val':
            batchSize = config.trainingParms['valBatchSize']

    oversampTurnFrac = None
    if genType=='train':
        oversampTurnFrac = config.trainingParms['oversampTurnFrac']

    # constraints
    useIMUData  = config.constraintParms['useIMU']
    useEpiRot   = config.constraintParms['useEpiRot']
    useEpiTrans = config.constraintParms['useEpiTrans']

    # normalization
    useNormImages = config.normalizationParms['useNormImages']
    useNormTruth  = config.normalizationParms['useNormTruth']
    useNormIMU    = config.normalizationParms['useNormIMU']
    useNormEpi    = config.normalizationParms['useNormEpi']


    # Files
    truthFilesDict = config.kittiPrepared['truth']
    imuFilesDict = config.kittiPrepared['imu']
    epiFilesDict = config.kittiPrepared['epipolar']
    normImageFilesDict = config.kittiNormalized['normImages']
    normDataFilesDict  = config.kittiNormalized['normData']
    if 'normEpi' in config.kittiNormalized:
        normEpiFilesDict = config.kittiNormalized['normEpi']


    # Split Indexes
    splitFilesDict = config.kittiPrepared['split']
    splitFile = config.getInputFiles(splitFilesDict)
    with h5py.File(splitFile, 'r') as f:
        if genType=='test':
            turnIdxs = np.array(f['testTurnIdxs'])
            nonTurnIdxs = np.array(f['testNonTurnIdxs'])
        elif genType=='val':
            turnIdxs = np.array(f['valTurnIdxs'])
            nonTurnIdxs = np.array(f['valNonTurnIdxs'])
        else:
            turnIdxs = np.array(f['trainTurnIdxs'])
            nonTurnIdxs = np.array(f['trainNonTurnIdxs'])



    if useNormImages:
        # Image Files - Normalized
        firstImageNames = np.empty(0)
        secondImageNames = np.empty(0)
        for cam in usedCams:
            for seq in usedSeqs:
                imageNames = config.getInputFiles(normImageFilesDict, seq, cam)
                firstImageNames = np.append(firstImageNames, imageNames[:-1], axis=0)
                secondImageNames = np.append(secondImageNames, imageNames[1:], axis=0)

    # Get Truth Data
    truthData = np.empty((0,7))
    if useNormTruth:
        # Normalized
        normDataFile = config.getInputFiles(normDataFilesDict)
        with h5py.File(normDataFile, 'r') as f:
            norm_rot_xyz = np.array(f['rot_xyz'])
            norm_trans_xyz = np.array(f['trans_xyz'])
            norm_trans_rtp = np.array(f['trans_rtp'])
        norm_trans_scale = norm_trans_rtp[:, 0:1]
        norm_rts = np.concatenate((norm_rot_xyz, norm_trans_xyz, norm_trans_scale), axis=1)
        truthData = np.append(truthData, norm_rts, axis=0)
    else:
        # Original
        for seq in usedSeqs:
            truthFile = config.getInputFiles(truthFilesDict, seq)
            with h5py.File(truthFile, 'r') as f:
                rot_xyz = np.array(f['rot_xyz'])
                trans_xyz = np.array(f['trans_xyz'])
                trans_rtp = np.array(f['trans_rtp'])
            trans_scale = trans_rtp[:, 0:1]
            rts = np.concatenate((rot_xyz, trans_xyz, trans_scale), axis=1)
            truthData = np.append(truthData, rts, axis=0)

    if (numOut == 1):
        # get magnitude
        truthData = truthData[:, -1:]
    elif (numOut == 3):
        # get cartesian translation
        truthData = truthData[:, -4:-1]
    else:  # (numOut == 6)
        # get rotation and cartesian translation
        truthData = truthData[:, :-1]


    imuData = None
    if useIMUData:
        # Get IMU Data
        if useNormIMU:
            normDataFile = config.getInputFiles(normDataFilesDict)
            # Normalized
            imuData = np.empty((0, 3))
            with h5py.File(normDataFile, 'r') as f:
                norm_imu_rot = np.array(f['noisy_rot_xyz'])
            imuData = np.append(imuData, norm_imu_rot, axis=0)

            if imu_bias_error_dph is not None:
                # TODO: Figure out bias error units and calculation
                # sampleRate = config.thesisKittiParms['sampleRate']
                # imu_bias_error = imu_bias_error_dph * np.pi/180 * 1/3600 * 1/sampleRate # b (deg/hr) * pi/180 (rad/deg) * 1/3600 (hr/sec) * sec = b*pi/180*1/3600 (rad)

                # # Apply Error
                # normParmsFilesDict = config.kittiNormalized['normParms']
                # normParmsFile = config.getInputFiles(normParmsFilesDict)
                # imu_rot_parms = loadNormParms(normParmsFile, 'noisy_rot_xyz')

                # imuData_unnorm = undoNorm(imuData, imu_rot_parms) # denormalize
                # imuData_biased = imuData_unnorm + imu_bias_error # add error
                # imuData_biased_norm = applyNorm(imuData_biased, imu_rot_parms) # renormalize
                # imuData = imuData_biased_norm




                # Calculate sensor error values
                sampleRate = config.thesisKittiParms['sampleRate']
                imu_bias_error = imu_bias_error_dph * np.pi / 180 * 1 / 3600 * 1 / sampleRate  # b (deg/hr) * pi/180 (rad/deg) * 1/3600 (hr/sec) * sec = b*pi/180*1/3600 (rad)

                # Get truth data
                normDataFile = config.getInputFiles(normDataFilesDict)
                with h5py.File(normDataFile, 'r') as f:
                    norm_rot_xyz = np.array(f['rot_xyz'])

                # Get normalization parameters
                normParmsFilesDict = config.kittiNormalized['normParms']
                normParmsFile = config.getInputFiles(normParmsFilesDict)
                truth_rot_parms = loadNormParms(normParmsFile, 'rot_xyz')

                rot_xyz = undoNorm(norm_rot_xyz, truth_rot_parms)  # denormalize

                # Add sensor error to truth data rotations based on error values given
                imuData_biased = rot_xyz + imu_bias_error  # add error

                # Renormalize data based on IMU parameters
                imu_rot_parms = loadNormParms(normParmsFile, 'noisy_rot_xyz')
                imuData_biased_norm = applyNorm(imuData_biased, imu_rot_parms)  # renormalize
                imuData = imuData_biased_norm





            if imu_sensor_error_dpsh is not None:
                # Calculate sensor error values
                sampleRate = config.thesisKittiParms['sampleRate']

                angular_std = np.sqrt(imu_sensor_error_dpsh**2 * 1/3600 * 1/sampleRate) * np.pi/180  # radians

                # Get truth data
                normDataFile = config.getInputFiles(normDataFilesDict)
                with h5py.File(normDataFile, 'r') as f:
                    norm_rot_xyz = np.array(f['rot_xyz'])

                # Get normalization parameters
                normParmsFilesDict = config.kittiNormalized['normParms']
                normParmsFile = config.getInputFiles(normParmsFilesDict)
                truth_rot_parms = loadNormParms(normParmsFile, 'rot_xyz')

                # Denormalize data based on truth parameters
                rot_xyz = undoNorm(norm_rot_xyz, truth_rot_parms)  # denormalize

                # Add sensor error to truth data rotations based on error values given
                np.random.seed(1)
                noise_xyz = np.random.randn(rot_xyz.shape[0], rot_xyz.shape[1])
                imuData_noisy = rot_xyz + angular_std * noise_xyz

                # Renormalize data based on IMU parameters
                imu_rot_parms = loadNormParms(normParmsFile, 'noisy_rot_xyz')
                imuData_noisy_norm = applyNorm(imuData_noisy, imu_rot_parms) # renormalize
                imuData = imuData_noisy_norm

        else:
            # Original
            imuData = np.empty((0, 3))
            for seq in usedSeqs:
                imuFile = config.getInputFiles(imuFilesDict, seq)
                with h5py.File(imuFile, 'r') as f:
                    noisy_rot_xyz = np.array(f['noisy_rot_xyz'])
                imuData = np.append(imuData, noisy_rot_xyz, axis=0)



    epiRotData = None
    if useEpiRot:
        # Get Epipolar Rotation Data
        epiRotData = np.empty((0,3))
        if useNormEpi:
            # Normalized
            normEpiFile = config.getInputFiles(normEpiFilesDict)
            with h5py.File(normEpiFile, 'r') as f:
                norm_epi_rot = np.array(f['epi_rot_xyz'])
            epiRotData = np.append(epiRotData, norm_epi_rot, axis=0)
        else:
            # Original
            for cam in usedCams:
                for seq in usedSeqs:
                    epiFile = config.getInputFiles(epiFilesDict, seq, cam)
                    with h5py.File(epiFile, 'r') as f:
                        epi_rot_xyz = np.array(f['epi_rot_xyz'])
                    epiRotData = np.append(epiRotData, epi_rot_xyz, axis=0)


    epiTransData = None
    if useEpiTrans:
        # Get Epipolar Translation Data
        epiTransData = np.empty((0,3))
        if useNormEpi:
            # Normalized
            normEpiFile = config.getInputFiles(normEpiFilesDict)
            with h5py.File(normEpiFile, 'r') as f:
                norm_epi_trans = np.array(f['epi_trans_xyz'])
            epiTransData = np.append(epiTransData, norm_epi_trans, axis=0)
        else:
            # Original
            for cam in usedCams:
                for seq in usedSeqs:
                    epiFile = config.getInputFiles(epiFilesDict, seq, cam)
                    with h5py.File(epiFile, 'r') as f:
                        epi_trans_xyz = np.array(f['epi_trans_xyz'])
                    epiTransData = np.append(epiTransData, epi_trans_xyz, axis=0)


    generator = DataGenerator(configData=config,
                                    turn_idxs=turnIdxs,
                                    nonturn_idxs=nonTurnIdxs,
                                    prev_img_files=firstImageNames,
                                    next_img_files=secondImageNames,
                                    labels=truthData,
                                    frac_turn=oversampTurnFrac,
                                    imu_xyz=imuData,
                                    epi_rot=epiRotData,
                                    epi_trans=epiTransData,
                                    batch_size=batchSize,
                                    img_dim=targetImgSize,
                                    n_channels=numChan,
                                    shuffle=shuffleData)

    return (generator)