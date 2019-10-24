import numpy as np
import h5py
import os
import sys
from src.helpers.cfg import ThesisConfig
from src.helpers.kitti_helpers import getPoses
from src.helpers.coframes import cvtToRel, cvtToAbs, getRT_sd_ss2d, \
    getXYZ_ss2d, cvtToRpy_sd, cart2sph, sph2cart



# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)

# Parameters
kittiSeqs       = config.kittiSeqs
usedCams        = config.usedCams
recalcTruthData = config.expKittiParms['prepared']['recalcTruth']

# Files
poseFilesDict  = config.kittiOriginal['truth']
truthFilesDict = config.kittiPrepared['truth']


print()
print('v Creating Truth Data')
for cam in usedCams:
    for seqStr in kittiSeqs:
        seq = int(seqStr)
        print('%sReading Camera %s, Sequence %02d' % (' '*2, cam, seq))

        poseFile = config.getInputFiles(poseFilesDict, seq)
        if (poseFile is None):
            continue

        truthFile = config.getOutputFiles(truthFilesDict, recalcTruthData, seq)
        if (truthFile is None):
            continue

        # Get true R and t from pose files

        #   Row i represents the ith pose of the left camera coordinate
        # system (z pointing forwards) via a 3x4 transformation matrix.
        # The matricies take a point in the ith coordinate system and project
        # it to the 0th coordinate system. The translational part corresponds
        # to the pose of the left camera in the ith frame with respect to the
        # 0th frame.

        # T_0_i = [  R_0_i , t_0_0toi ]
        #         [ 0_(1x3),     1    ]

        # Import Truth Poses
        poses = getPoses(poseFile)

        T_o_i = np.zeros((len(poses),4,4))
        T_o_i[:,:3,:4], T_o_i[:,3,3] = poses, np.ones(len(poses))

        # Convert to Relative Frame
        T_ip1_i = cvtToRel(T_o_i)

        # Get Rotation and Translation
        R_ip1_i, t_ip1_ip12i = getRT_sd_ss2d(T_ip1_i)

        # Rotation: DCM to Rotation Vector
        # Rotation angles are in radians
        # Order: (In Camera Frame) RotX, RotY, RotZ
        rot_xyz = cvtToRpy_sd(R_ip1_i)[:,0,:]

        # Translation: Cartesian
        # Order: (In Camera Frame) TraX, TraY, TraZ
        trans_xyz = t_ip1_ip12i[:,:,0]

        # Translation: Spherical
        # Angles are in radians and magnitude is in meters
        # Order: (In Camera Frame) TraR, TraT, TraP
        trans_rtp = cart2sph(t_ip1_ip12i)[:,:,0]

        # Save Labels to h5py file
        with h5py.File(truthFile, 'w') as f:
            os.chmod(truthFile, 0o666)
            f.create_dataset('rot_xyz', data=rot_xyz)
            f.create_dataset('trans_xyz', data=trans_xyz)
            f.create_dataset('trans_rtp', data=trans_rtp)

print('^ Truth Data Created')
print()