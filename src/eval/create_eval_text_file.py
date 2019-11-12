from src.helpers.cfg import ThesisConfig
from keras.models import load_model
from src.helpers.custom_loss import scaledMSE_RT
import keras
from src.helpers.dataGenerator import DataGenerator

from src.helpers.training_helpers import getGenerator
import numpy as np
from src.helpers.helper_functions import undoNorm
import h5py
import os

from src.helpers.coframes import cvtToRel, cvtToAbs, getRT_sd_ss2d, \
    getXYZ_ss2d, cvtToRpy_sd, cvtToDcm_sd, cart2sph, sph2cart, getT_s_d


# select test sequences
testCam = 0
testSeq = 10

# select model to use
configFile = 'exp_configs/scale_test_2.yaml'



config = ThesisConfig(configFile)

# Parameters
numOutputs      = config.modelParms['numOutputs']
defaultLossFunc = config.modelParms['defaultLossFunction']
lossRotScale    = config.modelParms['lossRotScale']
useIMUData  = config.constraintParms['useIMU']
targetImageSize = config.thesisKittiParms['downsampledImageShape']
numChannels   = config.modelParms['numImageChannels']

# Get Files
checkpointFilesDict = config.trainPaths['checkpoint']
normParmsFilesDict = config.trainPaths['normParms']
truthFilesDict = config.kittiPrepared['truth']
imuFilesDict = config.kittiPrepared['imu']
epiFilesDict = config.kittiPrepared['epipolar']
normImageFilesDict = config.kittiNormalized['normImages']
normDataFilesDict  = config.kittiNormalized['normData']

history_filename = config.trainingParms['histFilename']
historyFilesDict = config.trainPaths['history']
history_filepath = config.getInputFiles(historyFilesDict)

with h5py.File(history_filepath, 'r') as f:
    epochs = np.array(f['epochs'], dtype=np.int)
    numEpochs = len(epochs)
    if 'val_loss' in f:
        val_loss = np.array(f['val_loss'])
        min_val_loss = np.min(val_loss)
        min_val_loss_epoch = (epochs[np.argmin(val_loss)])
        print('Min Validation Loss: %s, Epoch %s' % (min_val_loss, min_val_loss_epoch))

# Loss Function
if numOutputs>3:
    lossFunc = scaledMSE_RT(lossRotScale)
    keras.losses.lossFunction = lossFunc
else:
    lossFunc = defaultLossFunc


# Model Epoch Selection
checkpointFiles = config.getInputFiles(checkpointFilesDict)
checkpointFolder = config.getFolderRef(checkpointFiles)
checkpointFile = 'epoch_%03d%s' % (min_val_loss_epoch, checkpointFilesDict['type'])
modelFile = os.path.join(checkpointFolder, checkpointFile)
model = load_model(modelFile)

# get filenames
firstImageNames = np.empty(0)
secondImageNames = np.empty(0)
imageNames = config.getInputFiles(normImageFilesDict, testSeq, testCam)
firstImageNames = np.append(firstImageNames, imageNames[:-1], axis=0)
secondImageNames = np.append(secondImageNames, imageNames[1:], axis=0)

# get truth data
truthData = np.empty((0,7))
# Normalized
normDataFile = config.getInputFiles(normDataFilesDict)
with h5py.File(normDataFile, 'r') as f:
    norm_rot_xyz = np.array(f['rot_xyz'])
    norm_trans_xyz = np.array(f['trans_xyz'])
    norm_trans_rtp = np.array(f['trans_rtp'])
norm_trans_scale = norm_trans_rtp[:, 0:1]
norm_rts = np.concatenate((norm_rot_xyz, norm_trans_xyz, norm_trans_scale), axis=1)
truthData = np.append(truthData, norm_rts, axis=0)

if (numOutputs == 1):
    # get magnitude
    truthData = truthData[:, -1:]
elif (numOutputs == 3):
    # get cartesian translation
    truthData = truthData[:, -4:-1]
else:  # (numOut == 6)
    # get rotation and cartesian translation
    truthData = truthData[:, :-1]

imuData = None
if useIMUData:
    normDataFile = config.getInputFiles(normDataFilesDict)
    # Normalized
    imuData = np.empty((0, 3))
    with h5py.File(normDataFile, 'r') as f:
        norm_imu_rot = np.array(f['noisy_rot_xyz'])
    imuData = np.append(imuData, norm_imu_rot, axis=0)

# TODO: Add epipolar data



# get non-shuffling generator for each test sequence
imgPairIdxs = list(range(len(firstImageNames)))
testSeqDataGen = DataGenerator(configData=config, turn_idxs=imgPairIdxs[:2], nonturn_idxs=imgPairIdxs[2:],
              prev_img_files=firstImageNames, next_img_files=secondImageNames, labels=truthData,
              frac_turn=None, imu_xyz=imuData, epi_RT=None, batch_size=1, img_dim=targetImageSize,
              n_channels=numChannels, shuffle=False)







# get predictions
y_pred_norm = np.empty((0,numOutputs))
total = len(testSeqDataGen)
for idx, data in enumerate(testSeqDataGen):
    percentComplete = int(idx / total * 100)
    if divmod(idx, 300)[1] == 0:
        print('Percent Complete: %d%%' % percentComplete)

    x, y = data
    y_pred_norm = np.append(y_pred_norm, model.predict(x), axis=0)



# denormalize predictions
normParmsFile = config.getInputFiles(normParmsFilesDict)
with h5py.File(normParmsFile, 'r') as f:
    truth_rot_parms = np.array(f['rot_xyz'])
    truth_xyz_parms = np.array(f['trans_xyz'])
    truth_polar_parms = np.array(f['trans_rtp'])

if (numOutputs == 1):
    norm_parms = truth_polar_parms[:,0:1]
elif (numOutputs == 3):
    norm_parms = truth_xyz_parms
else:
    norm_parms = np.concatenate((truth_rot_parms,truth_xyz_parms), axis=1)

y_pred_real = undoNorm(y_pred_norm, norm_parms)








# convert predictions to pose matrix format

# get truth data
truthFile = config.getInputFiles(truthFilesDict, testSeq)
with h5py.File(truthFile, 'r') as f:
    rot_xyz_true = np.array(f['rot_xyz'])
    trans_xyz_true = np.array(f['trans_xyz'])
    trans_rtp_true = np.array(f['trans_rtp'])





if numOutputs==1:
    rot_xyz = rot_xyz_true
    trans_rtp = np.concatenate((y_pred_real,trans_rtp_true[:,1:]),axis=-1)
    trans_xyz = sph2cart(trans_rtp)
elif numOutputs==3:
    rot_xyz = rot_xyz_true
    trans_xyz = y_pred_real
else:
    rot_xyz = y_pred_real[:,:3]
    trans_xyz = y_pred_real[:,3:]

R_ip1_i = cvtToDcm_sd(rot_xyz)
t_ip1_ip12i = trans_xyz

T_ip1_i = getT_s_d(R_ip1_i, t_ip1_ip12i)
T_o_i = cvtToAbs(T_ip1_i)

poses = T_o_i[:,:3,:4]
poses = poses.reshape(len(poses),12)



# save to file
evalFilesDict = config.resultPaths['evaluations']
evalFiles = config.getOutputFiles(evalFilesDict, True)
evalFolder = config.getFolderRef(evalFiles)
saveFile = os.path.join(evalFolder,'%s_pred.txt' % testSeq)

np.savetxt(saveFile,poses,fmt='%.18e',delimiter=' ')
print('Saved File to: %s' % saveFile)

# python evaluation.py --result_dir=/home/jwatson27/thesis/results/CNN_test_9/evals --eva_seqs=10_pred