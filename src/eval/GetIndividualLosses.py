from src.helpers.cfg import ThesisConfig
from keras.models import load_model
from src.helpers.custom_loss import scaledMSE_RT
import keras
from src.helpers.training_helpers import getGenerator
import numpy as np
from src.helpers.helper_functions import undoNorm
import h5py
import os

# Determine model to use for evaluation
configFile = 'exp_configs/CNN_test_9.yaml'
evalType = 'val'
# evalType = 'test'




config = ThesisConfig(configFile)



# Parameters
numOutputs      = config.modelParms['numOutputs']
defaultLossFunc = config.modelParms['defaultLossFunction']
lossRotScale    = config.modelParms['lossRotScale']
targetImageSize = config.thesisKittiParms['downsampledImageShape']
numChannels   = config.modelParms['numImageChannels']

# Get Files
checkpointFilesDict = config.trainPaths['checkpoint']
normParmsFilesDict = config.trainPaths['normParms']

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


# Get data generator
evalGen = getGenerator(config, numOutputs, targetImageSize, numChannels, evalType, batchSize=1)



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

y_true_real = undoNorm(y_true_norm, norm_parms)
y_pred_real = undoNorm(y_pred_norm, norm_parms)



# Calculate average loss in each direction

avgRealLoss = np.sqrt(np.mean((y_true_real-y_pred_real)**2,axis=0)) # dx,dy,dz,X,Y,Z
# meanRealErr = np.mean(y_true_real-y_pred_real,axis=0)
stdRealErr = np.std(y_true_real-y_pred_real,axis=0)

if numOutputs==6:
    rotErr, transErr_meters = stdRealErr[:3], stdRealErr[3:]
    rotErr_deg = rotErr*180/np.pi
    rotErr_meters = rotErr*transErr_meters[-1]

    print('Rotation Error (meters):    dx: %8f, dy: %8f, dz: %8f' % (rotErr_meters[0], rotErr_meters[1], rotErr_meters[2]))
    print('Translation Error (meters):  x: %8f,  y: %8f,  z: %8f' % (transErr_meters[0], transErr_meters[1], transErr_meters[2]))
elif numOutputs==3:
    transErr_meters = stdRealErr
    print('Translation Error (meters):  x: %8f,  y: %8f,  z: %8f' % (transErr_meters[0], transErr_meters[1], transErr_meters[2]))
else:
    scaleErr_meters = stdRealErr
    print('Scale Error (meters): s: %8f' % (scaleErr_meters))




