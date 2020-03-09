import os
import sys
import numpy as np

from keras.models import load_model, model_from_yaml
from keras.utils import plot_model

from src.arch.VO_Models import buildModel
from src.helpers.cfg import ThesisConfig
from src.helpers.helper_functions import getOptimizer
from src.helpers.training_helpers import getCallbacksList, getGenerator, getTrainAndValGenerators
from src.helpers.custom_loss import scaledMSE_RT

import tensorflow as tf
from keras import backend as K
import keras.losses


# Load Configuration
configFile = 'exp_configs/scale_test_4.yaml'
config = ThesisConfig(configFile)

checkpoint_filename = config.checkpointParms['filename']
checkpointFilesDict = config.trainPaths['checkpoint']
checkpointFiles = config.getOutputFiles(checkpointFilesDict, True)
saveCheckpointPath = config.getFolderRef(checkpointFiles)
checkpoint_filepath = os.path.join(saveCheckpointPath, checkpoint_filename)

numOutputs    = config.modelParms['numOutputs']
lossRotScale = config.modelParms['lossRotScale']
defaultLossFunc = config.modelParms['defaultLossFunction']
if numOutputs>3:
    lossFunc = scaledMSE_RT(lossRotScale)
    keras.losses.lossFunction = lossFunc
else:
    lossFunc = defaultLossFunc

modelPath = checkpoint_filepath.format(epoch=1)
model = load_model(modelPath)

model.summary()
model.layers[1].summary()