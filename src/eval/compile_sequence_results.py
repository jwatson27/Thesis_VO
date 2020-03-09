from src.helpers.cfg import ThesisConfig
import numpy as np
import h5py
import os
from src.helpers.coframes import getT_s_d, cvtToDcm_sd, sph2cart, cvtToAbs, getRT_sd_ss2d, getXYZ_ss2d




y_preds = []
y_truth = []
file_truth = []
rot_truths = []
trans_truths = []
truths = []
rot_preds = []
trans_preds = []
preds = []


# tests = ['CNN_test_12', 'CNN_test_17']
# tests = ['trans_test_2', 'trans_test_4']
# tests = ['scale_test_2', 'scale_test_4']
tests = ['CNN_test_12', 'CNN_test_17','trans_test_2', 'trans_test_4','scale_test_2', 'scale_test_4']
evalType = 'test'
seq = 10



# test = tests[0]
# if True:
for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    # name = config.expName
    numOutputs = config.modelParms['numOutputs']

    evalFilesDict = config.resultPaths['evaluations']
    truthFilesDict = config.kittiPrepared['truth']
    # figFilesDict = config.resultPaths['figures']
    # figFolder = config.getFolderRef(config.getOutputFiles(figFilesDict, True))

    # Get Predictions Save File
    evalFolder = ''
    for pathSection in evalFilesDict['dir']:
        evalFolder = os.path.join(evalFolder, pathSection)
    predictionsFile = os.path.join(evalFolder, 'seq10_'+evalType+'_predictions.hdf5')


    if os.path.exists(predictionsFile):

        # Load Predictions
        with h5py.File(predictionsFile, 'r') as f:
            y_pred_real = np.array(f['predictions'])
            y_true_real = np.array(f['truth'])
            evalType = str(np.array(f['evalType']))
            min_val_epoch = np.array(f['epoch'])
            min_val_loss = np.array(f['valLossAtEpoch'])

        y_preds.append(y_pred_real)
        y_truth.append(y_true_real)



    # CONVERT PREDICTIONS TO SE3

    # Truth Data
    truthFile = config.getInputFiles(truthFilesDict, seq)
    with h5py.File(truthFile, 'r') as f:
        true_rot_xyz = np.array(f['rot_xyz'])
        true_t_ip1_ip12i = np.array(f['trans_xyz'])
        true_trans_rtp = np.array(f['trans_rtp'])
    if numOutputs==1:
        file_truth.append(true_trans_rtp[:,0])
    elif numOutputs==3:
        file_truth.append(true_t_ip1_ip12i)
    else:
        file_truth.append(np.concatenate((true_rot_xyz,true_t_ip1_ip12i),axis=1))
    rot_truths.append(true_rot_xyz)
    trans_truths.append(true_t_ip1_ip12i)
    true_R_ip1_i = cvtToDcm_sd(true_rot_xyz)
    true_T_ip1_i = getT_s_d(true_R_ip1_i, true_t_ip1_ip12i)



    # Prediction Data
    pred_R_ip1_i = true_R_ip1_i
    if (numOutputs == 1):
        pred_trans_rtp = np.concatenate((y_pred_real,true_trans_rtp[:,1:]), axis=1)
        pred_t_ip1_ip12i = sph2cart(pred_trans_rtp)
    elif (numOutputs == 3):
        pred_t_ip1_ip12i = y_pred_real
    else:
        pred_rot_ip1_i, pred_t_ip1_ip12i = y_pred_real[:,:3], y_pred_real[:,3:]
        rot_preds.append(pred_rot_ip1_i)
        pred_R_ip1_i = cvtToDcm_sd(pred_rot_ip1_i)
    trans_preds.append(pred_t_ip1_ip12i)
    pred_T_ip1_i = getT_s_d(pred_R_ip1_i, pred_t_ip1_ip12i)



    # CONVERT TO ABSOLUTE
    true_T_o_i = cvtToAbs(true_T_ip1_i)
    pred_T_o_i = cvtToAbs(pred_T_ip1_i)


    with h5py.File(predictionsFile, 'a') as f:
        if 'true_T_o_i' not in f:
            f.create_dataset('true_T_o_i', data=true_T_o_i)
            print('Added true_T_o_i to %s' % predictionsFile)
        if 'pred_T_o_i' not in f:
            f.create_dataset('pred_T_o_i', data=pred_T_o_i)
            print('Added pred_T_o_i to %s' % predictionsFile)