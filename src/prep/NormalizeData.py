import numpy as np
from src.helpers.cfg import ThesisConfig
from src.helpers.helper_functions import applyNorm, undoNorm, loadNormParms, getH5pyData
import h5py
import sys



def calcRotScaleParm(rotData,transData,trainIdxs):
    trans = transData[trainIdxs, :]
    rot = rotData[trainIdxs, :]
    trans_means = np.mean(trans, axis=0)
    rot_means = np.mean(rot, axis=0)
    scaleParm = abs(np.mean(trans_means) / np.mean(rot_means))
    return scaleParm



# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)


# Parameters
usedCams = config.usedCams
usedSeqs = config.usedSeqs
reapplyDataNorm = config.expKittiParms['prepared']['runPrep']['reapplyDataNorm']

# Files
splitFilesDict = config.kittiPrepared['split']
truthFilesDict = config.kittiPrepared['truth']
imuFilesDict = config.kittiPrepared['imu']
epiFilesDict = config.kittiPrepared['epipolar']
normParmsFilesDict = config.kittiNormalized['normParms']
normDataFilesDict = config.kittiNormalized['normData']
normEpiFilesDict = config.kittiNormalized['normEpi']

print()
print('v Normalizing Data')



# Read in Normalization Parameters
normParmsFile = config.getInputFiles(normParmsFilesDict)

truth_rot_parms   = loadNormParms(normParmsFile, 'rot_xyz')
truth_xyz_parms   = loadNormParms(normParmsFile, 'trans_xyz')
truth_polar_parms = loadNormParms(normParmsFile, 'trans_rtp')
imu_rot_parms     = loadNormParms(normParmsFile, 'noisy_rot_xyz')
epi_rot_parms     = loadNormParms(normParmsFile, 'epi_rot_xyz')
epi_trans_parms   = loadNormParms(normParmsFile, 'epi_trans_xyz')




# Outputs

# Truth data - Rotation
truth_rot = getH5pyData(config, truthFilesDict, usedSeqs, 'rot_xyz', 3)
norm_truth_rot = applyNorm(truth_rot, truth_rot_parms)

# Truth data - Cartesian Translation
truth_xyz = getH5pyData(config, truthFilesDict, usedSeqs, 'trans_xyz', 3)
norm_truth_xyz = applyNorm(truth_xyz, truth_xyz_parms)

# Truth data - Polar Translation
truth_polar = getH5pyData(config, truthFilesDict, usedSeqs, 'trans_rtp', 3)
norm_truth_polar = applyNorm(truth_polar, truth_polar_parms)

# IMU data - Rotation
imu_rot = getH5pyData(config, imuFilesDict, usedSeqs, 'noisy_rot_xyz', 3)
norm_imu_rot = applyNorm(imu_rot, imu_rot_parms)

# Epipolar data - Rotation
epi_rot = np.empty((0,3))
for cam in usedCams:
    epi_rot_cam = getH5pyData(config, epiFilesDict, usedSeqs, 'epi_rot_xyz', 3, camera=cam)
    epi_rot = np.append(epi_rot, epi_rot_cam, axis=0)
norm_epi_rot = applyNorm(epi_rot, epi_rot_parms)

# Epipolar data - Translation
epi_trans = np.empty((0,3))
for cam in usedCams:
    epi_trans_cam = getH5pyData(config, epiFilesDict, usedSeqs, 'epi_trans_xyz', 3, camera=cam)
    epi_trans = np.append(epi_trans, epi_trans_cam, axis=0)
norm_epi_trans = applyNorm(epi_trans, epi_trans_parms)





# Save to File
normDataFile = config.getOutputFiles(normDataFilesDict,reapplyDataNorm)
if normDataFile is not None:
    print('Saving Normalized Truth and IMU Data')
    with h5py.File(normDataFile, 'w') as f:
        f.create_dataset('rot_xyz', data=norm_truth_rot)
        f.create_dataset('trans_xyz', data=norm_truth_xyz)
        f.create_dataset('trans_rtp', data=norm_truth_polar)
        f.create_dataset('noisy_rot_xyz', data=norm_imu_rot)

normEpiFile = config.getOutputFiles(normEpiFilesDict,reapplyDataNorm)
if normEpiFile is not None:
    print('Saving Normalized Epipolar Data')
    with h5py.File(normEpiFile, 'w') as f:
        f.create_dataset('epi_rot_xyz', data=norm_epi_rot)
        f.create_dataset('epi_trans_xyz', data=norm_epi_trans)


print('^ Data Normalization Complete')
print()
