from src.helpers.eval_helpers import getPredictions
from src.helpers.cfg import ThesisConfig
import os
import numpy as np
import sys
import keras.backend as K
import time
import logging


# IMU tests: CNN_test_12, trans_test_2, scale_test_2

# Add sensor errors to input data and get predictions
# Try range of sensor error values for each input axis

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


tests = ['CNN_test_12', 'trans_test_2', 'scale_test_2']
# sensorErrValues = [-3600, -1000, -100, -10, -1, -0.1, -0.01,
#                    0, 0.01, 0.1, 0.25, 0.5, 0.75,
#                    1, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 3600]
sensorErrValues = [0.0001, 0.001]

evalType = None
if len(sys.argv)>1:
    evalType = sys.argv[1]
if evalType is None:
    evalType = 'val'
evalType = 'test'

for test in tests:
    configFile = os.path.join('exp_configs', test + '.yaml')
    config = ThesisConfig(configFile)

    for err in sensorErrValues:
        t = time.time()
        K.clear_session()

        print('Getting predictions for IMU test %s with a sensor error of %s' % (tests, err))
        getPredictions(config, evalType, imu_sensor_dpsh=err, saveData=True);
        for axis in range(3):
            sensor_axes = np.zeros((1,3))
            sensor_axes[:,axis] = err
            print('\nGetting predictions for IMU test %s with a sensor error of %s' % (test, sensor_axes))
            getPredictions(config, evalType, imu_sensor_dpsh=sensor_axes, saveData=True);

        elapsed = time.time() - t
        print('time: %s' % (elapsed))
