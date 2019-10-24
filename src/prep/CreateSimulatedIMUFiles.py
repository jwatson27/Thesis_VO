import numpy as np
import h5py
import os
import sys
from src.helpers.cfg import ThesisConfig



# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)

# Parameters
kittiSeqs     = config.kittiSeqs
usedCams      = config.usedCams
recalcIMUData = config.expKittiParms['prepared']['runPrep']['recalcIMU']

# Files
truthFilesDict = config.kittiPrepared['truth']
imuFilesDict   = config.kittiPrepared['imu']


# Calculations
# deg^2/hr * 1/3600 * time_interval
sampleRate = config.thesisKittiParms['sampleRate'] # Hz
arwError = config.expKittiParms['imu']['arwError'] # deg/sqrt(hr) -- Based on Novatel UIMU-HG1700 # TODO: Cite novatel datasheet
angular_std = np.sqrt(arwError**2 * 1/3600 * 1/sampleRate) * np.pi/180 # radians
#                sqrt(  deg^2/hr   * hr/sec *      sec     ) * radians/deg
# VALUE IS APPROXIMATELY 0.002635 deg


print()
print('v Creating IMU Data')
for cam in usedCams:
    for seqStr in kittiSeqs:
        seq = int(seqStr)
        print('%sReading Camera %s, Sequence %02d' % (' '*2, cam, seq))

        truthFile = config.getInputFiles(truthFilesDict, seq)
        if (truthFile is None):
            continue

        imuFile = config.getOutputFiles(imuFilesDict, recalcIMUData, seq)
        if (imuFile is None):
            continue

        # Read in rotation truth poses
        with h5py.File(truthFile, 'r') as f:
            rot_xyz = np.array(f['rot_xyz'])

        # Add noise to rotation piece
        noise_xyz = np.random.randn(rot_xyz.shape[0], rot_xyz.shape[1])
        noisy_rot_xyz = rot_xyz + angular_std * noise_xyz

        # Save noisy rotation angles to h5py file
        with h5py.File(imuFile, 'w') as f:
            os.chmod(imuFile, 0o666)
            f.create_dataset('noisy_rot_xyz', data=noisy_rot_xyz)


print('^ IMU Data Created')
print()