from src.helpers.custom_callbacks import PlotHistory
from src.helpers.cfg import ThesisConfig
import os
import h5py
import numpy as np

# Load Configuration
# configFileList = ['exp_configs/CNN_test_4.yaml',
#                   'exp_configs/CNN_test_9.yaml',
#                   'exp_configs/CNN_test_6.yaml',
#                   'exp_configs/CNN_test_10.yaml']


# configFileList = ['exp_configs/scale_test_1.yaml',
#                   'exp_configs/scale_test_2.yaml',
#                   'exp_configs/scale_test_3.yaml']
#
# configFileList = ['exp_configs/trans_test_1.yaml',
#                   'exp_configs/trans_test_2.yaml',
#                   'exp_configs/trans_test_3.yaml']
#
# configFileList = ['exp_configs/CNN_test_11.yaml',
#                   'exp_configs/CNN_test_12.yaml',
#                   'exp_configs/CNN_test_13.yaml',
#                   'exp_configs/CNN_test_14.yaml',
#                   'exp_configs/CNN_test_15.yaml',
#                   'exp_configs/CNN_test_16.yaml',
#                   'exp_configs/CNN_test_17.yaml']

configFileList = ['exp_configs/scale_test_3.yaml']


for configFile in configFileList:
    config = ThesisConfig(configFile)

    history_filename = config.trainingParms['histFilename']
    historyFilesDict = config.trainPaths['history']
    historyFiles = config.getInputFiles(historyFilesDict)
    saveHistoryPath = config.getFolderRef(historyFiles)
    history_filepath = os.path.join(saveHistoryPath, history_filename)
    saveFigFilesDict = config.resultPaths['figures']
    figureFile = config.getOutputFiles(saveFigFilesDict, True)
    saveFigFolder = config.getFolderRef(figureFile)
    saveFigFilename = config.experiment['experiment']['name']+config.resultPaths['figures']['type']
    saveFigFile = os.path.join(saveFigFolder, saveFigFilename)

    if os.path.exists(history_filepath):
        print()
        with h5py.File(history_filepath, 'r') as f:
            epochs = np.array(f['epochs'], dtype=np.int)
            numEpochs = len(epochs)
            if 'loss' in f:
                train_loss = np.array(f['loss'])
                min_train_loss = np.min(train_loss)
                min_train_loss_epoch = (epochs[np.argmin(train_loss)])
                print('Min Training Loss: %s, Epoch %s' % (min_train_loss, min_train_loss_epoch))
            if 'val_loss' in f:
                val_loss = np.array(f['val_loss'])
                min_val_loss = np.min(val_loss)
                min_val_loss_epoch = (epochs[np.argmin(val_loss)])
                print('Min Validation Loss: %s, Epoch %s' % (min_val_loss, min_val_loss_epoch))
        ph = PlotHistory(history_filepath, blockProcessing=True, saveFigFile=saveFigFile)
        ph.on_epoch_end(numEpochs)
        print()
    else:
        print('History file does not exist for selected experiment')


    # datasets = {}
    # with h5py.File(history_filepath, 'r') as f:
    #     for ds in f:
    #         datasets[ds] = np.array(f[ds])
    #
    # epochs = datasets['epochs']
    # loss = datasets['loss']
    # val_loss = datasets['val_loss']
    #
    # epochs = epochs[:-1]
    # loss = loss[:-1]
    # val_loss = val_loss[:-1]
    #
    # epochs = np.array(epochs, dtype=np.int)
    #
    # with h5py.File(history_filepath, 'w') as f:
    #     f.create_dataset('epochs', data=epochs)
    #     f.create_dataset('loss', data=loss)
    #     f.create_dataset('val_loss', data=val_loss)
