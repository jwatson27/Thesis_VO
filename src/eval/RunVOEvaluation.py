from src.helpers.cfg import ThesisConfig
from src.helpers.eval_helpers import getPredictions
import os


# Determine model to use for evaluation
evalType = 'val'
# evalType = 'test'


tests = ['CNN_test_4', 'CNN_test_6', 'CNN_test_9', 'CNN_test_10']
# tests = ['CNN_test_17', 'trans_test_4', 'scale_test_4']
# tests = ['CNN_test_12', 'trans_test_2', 'scale_test_2']


# for test in tests:
for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    getPredictions(config, evalType, saveData=True, batchSize=1);

