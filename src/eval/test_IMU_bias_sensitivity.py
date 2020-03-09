from src.helpers.eval_helpers import getPredictions
from src.helpers.cfg import ThesisConfig
import os
import numpy as np
import sys
import keras.backend as K
import time
import logging

# IMU tests: CNN_test_12, trans_test_2, scale_test_2

# Add bias errors to input data and get predictions
# Try range of bias values for each input axis, start with 1 deg per hour bias

# Run a similar test for the guassian errors by varying the std value and getting predictions


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


tests = ['CNN_test_12', 'trans_test_2', 'scale_test_2']
# biasValues = [-3600, -1000, -100, -10, -1, 0, 0.25, 0.5, 0.75, 1, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 3600]
biasValues = [0, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 5,
              10, 50, 100, 200, 300, 400, 500, 600, 700, 800,
              900, 1000, 2000, 3000, 3600]

evalType = None
if len(sys.argv)>1:
    evalType = sys.argv[1]
if evalType is None:
    evalType = 'val'

for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    for bias in biasValues:
        t = time.time()
        K.clear_session()

        print('Getting predictions for IMU test %s with a bias of %s' % (tests, bias))
        getPredictions(config, evalType, imu_bias_dph=bias, saveData=True);
        for axis in range(3):
            bias_axes = np.zeros((1,3))
            bias_axes[:,axis] = bias
            print('\nGetting predictions for IMU test %s with a bias of %s' % (test, bias_axes))
            getPredictions(config, evalType, imu_bias_dph=bias_axes, saveData=True);

        elapsed = time.time() - t
        print('time: %s' % (elapsed))
