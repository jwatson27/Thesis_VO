

from src.helpers.eval_helpers import getPredictions
from src.helpers.cfg import ThesisConfig
import os
import numpy as np
import sys
import keras.backend as K
import logging
import time

from src.helpers.helper_functions import getValLoss




# tests = ['CNN_test_12', 'trans_test_2', 'scale_test_2']
# tests = ['CNN_test_4', 'CNN_test_6', 'CNN_test_9', 'CNN_test_10']
# tests = ['CNN_test_10', 'CNN_test_9', 'CNN_test_6']
tests = ['trans_test_2', 'trans_test_4', 'scale_test_2', 'scale_test_4']

# evalType = None
# if len(sys.argv)>1:
#     evalType = sys.argv[1]
# if evalType is None:
evalType = 'val'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# if True:
#     test = tests[0]
for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    checkpointFilesDict = config.trainPaths['checkpoint']
    checkpointFiles = config.getInputFiles(checkpointFilesDict)

    print('Getting all epoch predictions for %s' % (test))
    totalEpochs = len(checkpointFiles)
    for epoch in range(1,totalEpochs+1):
        t = time.time()
        K.clear_session()
        print('  epoch %03d of %03d' % (epoch, totalEpochs))
        getPredictions(config, evalType, epochNum=epoch, batchSize=1, saveData=True);
        elapsed = time.time() - t
        print('    time: %s' % (elapsed))




