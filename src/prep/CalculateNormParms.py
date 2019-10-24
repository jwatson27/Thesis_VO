import numpy as np
import os
import sys
import h5py


from src.helpers.cfg import ThesisConfig
from src.helpers.helper_functions import getNormParms, getH5pyData






def getTrainIdxs(configClass, splitFilesDict):
    splitFile = configClass.getInputFiles(splitFilesDict)
    if splitFile is None:
        print('Invalid Split File')
        return None
    with h5py.File(splitFile, 'r') as f:
        trainNonturn = np.array(f['trainNonTurnIdxs'])
        trainTurn = np.array(f['trainTurnIdxs'])
    trainIdxs = np.concatenate((trainNonturn, trainTurn))
    return trainIdxs


# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)

# Parameters
usedSeqs = config.usedSeqs
recalcDataNorm = config.expKittiParms['prepared']['recalcNormParms']


# Files
splitFilesDict = config.kittiPrepared['split']
truthFilesDict = config.kittiPrepared['truth']
imuFilesDict = config.kittiPrepared['imu']
epiFilesDict = config.kittiPrepared['epipolar']
normParmsFilesDict = config.trainPaths['normParms']



print()
print('v Calculating Normalization Parameters')

# Get Training Indexes from Split File

trainIdxs = getTrainIdxs(config, splitFilesDict)


# Get Normalization Parameters


# Truth data - Rotation
truth_rot = getH5pyData(config, truthFilesDict, usedSeqs, 'rot_xyz', 3)
rot_parms = getNormParms(truth_rot, trainIdxs)


# Truth data - Cartesian Translation
truth_trans = getH5pyData(config, truthFilesDict, usedSeqs, 'trans_xyz', 3)
trans_parms = getNormParms(truth_trans, trainIdxs)


# Truth data - Polar Translation
truth_polar = getH5pyData(config, truthFilesDict, usedSeqs, 'trans_rtp', 3)
polar_parms = getNormParms(truth_polar, trainIdxs)


# IMU data - Rotation
imu_rot = getH5pyData(config, imuFilesDict, usedSeqs, 'noisy_rot_xyz', 3)
imu_parms = getNormParms(imu_rot, trainIdxs)


# TODO: Epipolar data - Rotation
# epi_rot = getH5pyData(config, epiFilesDict, usedSeqs, 'epi_rot_xyz', 3)
# epi_rot_parms = getNormParms(epi_rot, trainIdxs)


# TODO: Epipolar data - Translation
# epi_trans = getH5pyData(config, epiFilesDict, usedSeqs, 'epi_trans_xyz', 3)
# epi_trans_parms = getNormParms(epi_trans, trainIdxs)


# Save to File

normParmsFile = config.getOutputFiles(normParmsFilesDict, recalcDataNorm)
if normParmsFile is None:
    sys.exit()

with h5py.File(normParmsFile, 'w') as f:

    f.create_dataset('rot_xyz', data=rot_parms)
    f.create_dataset('trans_xyz', data=trans_parms)
    f.create_dataset('trans_rtp', data=polar_parms)
    f.create_dataset('noisy_rot_xyz', data=imu_parms)
    # TODO: Add Epipolar Data
    # f.create_dataset('epi_rot_xyz', data=epi_rot_parms)
    # f.create_dataset('epi_trans_xyz', data=epi_trans_parms)


print('^ Normalization Parameters Calculated')
print()
