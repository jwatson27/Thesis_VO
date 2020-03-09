from src.helpers.cfg import ThesisConfig
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from src.helpers.coframes import getT_s_d, cvtToDcm_sd, sph2cart, cvtToAbs, getRT_sd_ss2d, getXYZ_ss2d



def plotBirdsEye(T_o_i, label=None):
    _, t_o_o2i = getRT_sd_ss2d(T_o_i)
    x, _, z = getXYZ_ss2d(t_o_o2i)
    plt.plot(x, z, label=label)





# tests = ['CNN_test_12', 'CNN_test_17']
# # tests = ['trans_test_2', 'trans_test_4']
# # tests = ['scale_test_2', 'scale_test_4']
# # tests = ['trans_test_2', 'trans_test_4', 'scale_test_2', 'scale_test_4']
# seq = 10
# evalType = 'test'




y_preds = []
y_truth = []
file_truth = []
rot_truths = []
trans_truths = []
truths = []
rot_preds = []
trans_preds = []
preds = []


tests = ['CNN_test_12', 'CNN_test_17']
# tests = ['trans_test_2', 'trans_test_4']
# tests = ['scale_test_2', 'scale_test_4']
# tests = ['CNN_test_12', 'CNN_test_17','trans_test_2', 'trans_test_4','scale_test_2', 'scale_test_4']
evalType = 'test'
seq = 10
labels = ['INS-Aided', 'Only Images']

# test = tests[0]
# if True:
for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    # name = config.expName
    numOutputs = config.modelParms['numOutputs']

    evalFilesDict = config.resultPaths['evaluations']
    truthFilesDict = config.kittiPrepared['truth']
    figFilesDict = config.resultPaths['figures']
    figFolder = config.getFolderRef(config.getOutputFiles(figFilesDict, True))

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

    truths.append(true_T_o_i)
    preds.append(pred_T_o_i)








# for i in range(numOutputs):
#     plt.figure()
#     plt.plot(y_truth[0][:, i], label=str(i) + '_truth_aided')
#     # plt.plot(y_truth[1][:, i], label=str(i) + '_truth_base')
#     plt.plot(y_preds[0][:, i], label=str(i) + '_pred_aided')
#     plt.plot(y_preds[1][:, i], label=str(i) + '_pred_base')
#     plt.legend()


plt.figure()
plotBirdsEye(truths[1], label='Truth')
for i, pred in enumerate(preds):
    plotBirdsEye(pred, label=labels[i])
plt.legend()


# for j in range(len(y_preds)):
#     y_pred_real = y_preds[j]
#     y_true_real = y_truth[j]
#     for i in range(y_pred_real.shape[1]):
#         plt.figure()
#         plt.plot(y_pred_real[:, i], label=str(i) + '_pred')
#         plt.plot(y_true_real[:, i], label=str(i) + '_true')
#         plt.title(tests[j])
#         plt.legend()


















# Plot Birds Eyefor k in range(3):
#     plt.figure()
#     plt.plot(truth_rots[0][:,k])
#     for i, pred_rot in enumerate(preds_rots):
#         plt.plot(pred_rot[:,k])


# plt.show(block=True)


# for k in range(3):
#     plt.figure()
#     plt.plot(truth_rots[0][:,k])
#     for i, pred_rot in enumerate(preds_rots):
#         plt.plot(pred_rot[:,k])
# # plt.show(block=True)

#
# for j in range(len(preds_rots)):
#     y_pred_real = preds[j]
#     y_true_real = truth[j]
#     for i in range(y_pred_real.shape[1]):
#         plt.figure()
#         plt.plot(y_pred_real[:, i], label=str(i)+'_pred')
#         plt.plot(y_true_real[:, i], label=str(i)+'_true')
#         plt.title(tests[j])
#         plt.legend()



plt.show(block=True)