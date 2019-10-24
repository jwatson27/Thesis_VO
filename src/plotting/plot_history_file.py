from src.helpers.custom_callbacks import PlotHistory
from src.helpers.cfg import ThesisConfig
import os
import h5py


# Load Configuration
configFile = 'exp_configs/CNN_test_0.yaml'
config = ThesisConfig(configFile)

history_filename = config.trainingParms['histFilename']
historyFilesDict = config.trainPaths['history']
historyFiles = config.getOutputFiles(historyFilesDict, True)
saveHistoryPath = config.getFolderRef(historyFiles)
history_filepath = os.path.join(saveHistoryPath, history_filename)

if os.path.exists(history_filepath):
    with h5py.File(history_filepath, 'r') as f:
        numEpochs = len(f['epochs'])
    ph = PlotHistory(history_filepath, blockProcessing=True)
    ph.on_epoch_end(numEpochs)
else:
    print('History file does not exist for selected experiment')