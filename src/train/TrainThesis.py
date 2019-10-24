import os
import sys
import numpy as np

from keras.models import load_model, model_from_yaml
from keras.utils import plot_model

from src.arch.VO_Models import buildDualHeadModel
from src.helpers.cfg import ThesisConfig
from src.helpers.helper_functions import getOptimizer
from src.helpers.training_helpers import getCallbacksList, getTrainAndValGenerators
from src.helpers.custom_loss import scaledMSE_RT



import tensorflow as tf
from keras import backend as K
import keras.losses

# Supress TensorFlow Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure GPU Space with TensorFlow
cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=cfg))







# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)


# Parameters
expName = config.experiment['experiment']['name']

# Dataset
targetImageSize = config.expKittiParms['prepared']['downsampledImageShape']

# Training
OVERWRITE_EPOCHS = config.trainingParms['overwriteTrainEpochs']
nextEpoch        = config.trainingParms['nextEpoch']
totalEpochs      = config.trainingParms['totalEpochs']
initLR           = config.trainingParms['initLearningRate']
trainVerbosity   = config.trainingParms['verbosity']

# Model
CNN_Name      = config.modelParms['CNN_name']
CNN_Dropout   = config.modelParms['dropout']['CNN']
VO_Dropout    = config.modelParms['dropout']['VO']
numChannels   = config.modelParms['numImageChannels']
numOutputs    = config.modelParms['numOutputs']
useIMUData    = config.constraintParms['useIMU']
useEpiRot     = config.constraintParms['useEpiRot']
useEpiTrans   = config.constraintParms['useEpiTrans']
defaultLossFunc = config.modelParms['defaultLossFunction']
lossRotScale = config.modelParms['lossRotScale']
optimizerType = config.modelParms['optimizerType']
model_yaml_filename = config.modelParms['modelYamlFilename']

# Callbacks
checkpoint_filename = config.checkpointParms['filename']
history_filename    = config.trainingParms['histFilename']


# Get Files

modelYamlsDict = config.trainPaths['model']
modelYamls = config.getOutputFiles(modelYamlsDict, True)
saveModelPath = config.getFolderRef(modelYamls)
model_filepath = os.path.join(saveModelPath, model_yaml_filename)

checkpointFilesDict = config.trainPaths['checkpoint']
checkpointFiles = config.getOutputFiles(checkpointFilesDict, True)
saveCheckpointPath = config.getFolderRef(checkpointFiles)
checkpoint_filepath = os.path.join(saveCheckpointPath, checkpoint_filename)

historyFilesDict = config.trainPaths['history']
historyFiles = config.getOutputFiles(historyFilesDict, True)
saveHistoryPath = config.getFolderRef(historyFiles)
history_filepath = os.path.join(saveHistoryPath, history_filename)

# Get Callbacks List
callbacksList = getCallbacksList(config, history_filepath, checkpoint_filepath)

# Create Generators
trainGen, valGen = getTrainAndValGenerators(config, numOutputs, targetImageSize, numChannels)


if numOutputs>3:
    lossFunc = scaledMSE_RT(lossRotScale)
    keras.losses.lossFunction = lossFunc
else:
    lossFunc = defaultLossFunc




imageShape = tuple(np.append(np.array(targetImageSize),numChannels))

folderExists = os.path.exists(saveCheckpointPath)

if isinstance(checkpointFiles, str):
    if not config.isFileRef(checkpointFiles):
        currEpoch = 1
    else:
        currEpoch = 2
else:
    currEpoch = len(checkpointFiles)+1
prevEpoch = nextEpoch-1
if (nextEpoch < 1) or (not folderExists):
    if not folderExists:
        os.makedirs(saveCheckpointPath)
    # Start/Continue Training (next epoch)
    nextEpoch = currEpoch
    prevEpoch = nextEpoch-1
elif (nextEpoch > currEpoch):
    # No previous model
    print('Cannot Start at Epoch %s because there is no Epoch %s. Exiting...' % (nextEpoch, prevEpoch))
    sys.exit()
elif (nextEpoch < currEpoch) and (not OVERWRITE_EPOCHS):
    # Overwrite is not allowed
    print('Cannot overwrite. Exiting...')
    sys.exit()


if (nextEpoch == 1):
    # Build Model
    model, cnn_model = buildDualHeadModel(imageShape,
                                          numOutputs,
                                          vo_dropout=VO_Dropout,
                                          cnn_dropout=CNN_Dropout,
                                          cnn_type=CNN_Name,
                                          include_imu=useIMUData)

    model.compile(optimizer=getOptimizer(optimizerType, lr=initLR), loss=lossFunc)

    # Save Yaml File for Model
    with open(model_filepath, 'w') as f:
        f.write(model.to_yaml())

else:
    # Load Old Model
    modelPath = checkpoint_filepath.format(epoch=prevEpoch)
    model = load_model(modelPath)


model.summary()










numIters = totalEpochs-nextEpoch+1
print()
if numIters==0:
    print('Model Fully Trained at %s Epoch. Exiting...' % (totalEpochs))
    sys.exit()
else:
    print('Training Model for %s epochs (Epochs %s to %s)' % (numIters, nextEpoch, totalEpochs))
print()

history = model.fit_generator(generator=trainGen,
                              epochs=totalEpochs,
                              verbose=trainVerbosity,
                              callbacks=callbacksList,
                              validation_data=valGen,
                              initial_epoch=prevEpoch)































# OLD TRAINING FUNCTION

# def train_net_gen(model, train_gen, val_gen=None, callbacks_list=None,
#                   reduceLRdict=None, start_iteration=1, training_iterations=6,
#                   modelName='CNN_model', save_all_model_iterations=True):
#
#     if val_gen is None:
#         noVal = True
#     else:
#         noVal = False
#
#     plt.figure(1)
#     plt.ion()
#     plt.show()
#     windowSize = 40
#     first = True
#
#     tz = 'US/Eastern'
#     start_time = datetime.now(timezone(tz))
#
#
#     save_dir = os.path.join(os.getcwd(), 'saved_models', modelName)
#     history_file_name = 'training_history.hdf5'
#     if not start_iteration == 1:
#         old_history = load_history(history_file_name, save_dir, noVal)
#         loss = list(old_history['loss'])
#         if not noVal:
#             val_loss = list(old_history['val_loss'])
#         lr_changes = list(old_history['lr_changes'])
#         if plt.fignum_exists(1):
#             startEpoch = np.where(np.array(loss)<1)[0][0]
#             plotLoss(hist=old_history, name=modelName, start=startEpoch)
#             if first:
#                 plt.legend()
#                 first = False
#             plt.draw()
#             if len(loss) > windowSize:
#                 ymax = round(np.max(loss[-windowSize:]), dir='up')
#                 ymin = round(np.min(loss[-windowSize:]), dir='down')
#                 if not noVal:
#                     valmax = round(np.max(val_loss[-windowSize:]), dir='up')
#                     valmin = round(np.min(val_loss[-windowSize:]), dir='down')
#                     if valmax > ymax:
#                         ymax = valmax
#                     if valmin < ymin:
#                         ymin = valmin
#                 plt.xlim(left=len(loss) - windowSize, right=len(loss))
#                 plt.ylim(bottom=ymin, top=ymax)
#             plt.pause(0.001)
#     else:
#         loss = []
#         val_loss = []
#         lr_changes = [1]
#
#     for training_iteration in range(start_iteration, start_iteration + training_iterations):
#         curr_time = datetime.now(timezone(tz))
#         elapsed = '%02d:%02d:%02d' % hms_time(curr_time - start_time)
#         curr_time = curr_time.strftime('%H:%M:%S')
#         print()
#         print('-' * 50)
#         print('Training Iteration (epoch) #: %s    Time: %s,  Elapsed: %s' % (training_iteration, curr_time, elapsed))
#
#         history = model.fit_generator(generator=train_gen, validation_data=val_gen,
#                                       epochs=1, verbose=2, callbacks=callbacks_list)
#
#         loss.append(history.history['loss'][0])
#         if not noVal:
#             val_loss.append(history.history['val_loss'][0])
#
#         sleep(0.1)  # https://github.com/fchollet/keras/issues/2110
#
#         sys.stdout.flush()
#
#         history = {}
#         history['loss'] = np.array(loss)
#         if not noVal:
#             history['val_loss'] = np.array(val_loss)
#
#         if not noVal:
#             if reduceLRdict:
#                 print('Current LR: %s' % K.eval(model.optimizer.lr))
#                 monitor = reduceLRdict['monitor']
#                 patience = reduceLRdict['patience']
#                 hist = history[monitor]
#                 last_lr_change = lr_changes[-1]
#                 epochs_since_last_change = training_iteration - last_lr_change
#                 print('epochs_since_last_change: %s' % epochs_since_last_change)
#                 if epochs_since_last_change >= patience:
#                     # print('Previous %s: %s, Most Recent %s: %s' % (monitor, hist[-(patience+1)], monitor, hist[-1]))
#                     if abs(hist[-(patience+1)] - hist[-1]) < reduceLRdict['min_delta']:
#                         old_LR = K.eval(model.optimizer.lr)
#                         new_LR = old_LR*reduceLRdict['factor']
#                         lossFunc = model.loss_functions[0]
#                         model.compile(optimizer=optimizers.RMSprop(lr=new_LR), loss=lossFunc)
#                         lr_changes.append(training_iteration)
#                         print('Decreased LR from %s to %s' % (old_LR, new_LR))
#
#         history['lr_changes'] = np.array(lr_changes)
#         if plt.fignum_exists(1):
#             startEpoch = 1
#             plotLoss(hist=history, name=modelName, start=startEpoch)
#             if first and len(loss)>startEpoch:
#                 plt.legend()
#                 first = False
#             plt.draw()
#             if len(loss)>windowSize:
#                 ymax = round(np.max(loss[-windowSize:]), dir='up')
#                 ymin = round(np.min(loss[-windowSize:]), dir='down')
#                 if not noVal:
#                     valmax = round(np.max(val_loss[-windowSize:]), dir='up')
#                     valmin = round(np.min(val_loss[-windowSize:]), dir='down')
#                     if valmax > ymax:
#                         ymax = valmax
#                     if valmin < ymin:
#                         ymin = valmin
#                 plt.xlim(left=len(loss) - windowSize, right=len(loss))
#                 plt.ylim(bottom=ymin, top=ymax)
#             plt.pause(0.001)
#
#
#         print()
#
#         current_model_file_name = 'training_epoch_' + str(training_iteration) + '.h5'
#         if save_all_model_iterations:
#             save_model(model=model, save_dir=save_dir, model_file_name=current_model_file_name)
#         save_history(history, history_file_name, save_dir, noVal)
#
#     return model, history