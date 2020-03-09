from src.helpers.custom_loss import scaledMSE_RT
import keras
import keras.backend as K
from keras.models import load_model
from src.helpers.cfg import ThesisConfig
import numpy as np
import h5py
import os
import time
import logging

tests = ['CNN_test_4', 'CNN_test_6', 'CNN_test_9', 'CNN_test_10']
# tests = ['CNN_test_12','CNN_test_17','trans_test_2','trans_test_4','scale_test_2','scale_test_4']





# for test in tests:
#     configFile = os.path.join('exp_configs', test + '.yaml')
#     config = ThesisConfig(configFile)
#
#     numOutputs = config.modelParms['numOutputs']
#     lossRotScale    = config.modelParms['lossRotScale']
#     defaultLossFunc = config.modelParms['defaultLossFunction']
#
#     historyFilesDict = config.trainPaths['history']
#     historyFiles = config.getInputFiles(historyFilesDict)
#     historyFolder = config.getFolderRef(historyFiles)
#
#     lrFile = os.path.join(historyFolder, 'learning_rate_history.hdf5')
#
#     with h5py.File(lrFile, 'r') as f:
#         learning_rates = np.array(f['learning_rate'])
#
#     epochs = list(range(1,len(learning_rates)+1))
#
#     with h5py.File(lrFile, 'w') as f:
#         f.create_dataset('learning_rate', data=np.array(learning_rates))
#         f.create_dataset('epochs', data=np.array(epochs))
#
#     print('saved data to %s' % lrFile)




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# test = tests[0]
# if True:
#
for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    numOutputs = config.modelParms['numOutputs']
    lossRotScale    = config.modelParms['lossRotScale']
    defaultLossFunc = config.modelParms['defaultLossFunction']

    checkpointFilesDict = config.trainPaths['checkpoint']
    checkpointFiles = config.getInputFiles(checkpointFilesDict)
    if isinstance(checkpointFiles, str):
        checkpointFiles = np.array([checkpointFiles])
    checkpointFolder = config.getFolderRef(checkpointFiles)

    # Loss Function
    if numOutputs > 3:
        lossFunc = scaledMSE_RT(lossRotScale)
        keras.losses.lossFunction = lossFunc
    else:
        lossFunc = defaultLossFunc


    learning_rates = []
    epochs = []
    # FOR EACH MODEL
    totalEpochs = len(checkpointFiles)
    for modelFile in checkpointFiles:
        t = time.time()
        K.clear_session()

        # READ IN THE MODEL
        model = load_model(modelFile)

        # EXTRACT THE LEARNING RATE
        currLR = K.eval(model.optimizer.lr)

        epochStr = modelFile.rsplit('/', 1)[1].split('.')[0].split('_')[1]

        # STORE LEARNING RATE IN ARRAY
        epochs.append(int(epochStr))
        learning_rates.append(currLR)
        elapsed = time.time()-t

        print('epoch %s of %s, lr: %s, time: %s' % (epochStr, totalEpochs, currLR, elapsed))


    # GET HISTORY FOLDER
    historyFilesDict = config.trainPaths['history']
    historyFiles = config.getInputFiles(historyFilesDict)
    historyFolder = config.getFolderRef(historyFiles)

    lrFile = os.path.join(historyFolder, 'learning_rate_history.hdf5')

    epochs = list(epochs)

    # SAVE ARRAY TO FILE
    with h5py.File(lrFile, 'w') as f:
        f.create_dataset('learning_rate', data=np.array(learning_rates))
        f.create_dataset('epochs', data=np.array(epochs))

    print('saved data to %s' % lrFile)