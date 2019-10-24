# Helper functions for input/output related to the Kitti dataset

import numpy as np

def readCalFile(file: str):
    '''
    Function that reads lines of kitti cal file to dictionary

    :param file: kitti camera calibration filename
    :return: dictionary of flattened 3x4 camera cal matrices
    '''
    data = {}
    with open(file, 'r') as cf:
        for line in cf.readlines():
            key, value = line.split(':', 1)

            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def getCalMat(calfile: str, camnum: int):
    '''
    Function that returns camera cal matrix for specified camera given
    kitti cal file

    :param calfile: kitti camera calibration filename
    :param camnum: kitti camera number 0-1 (gray), 2-3 (color)
    :return: numpy 3x3 calibration matrix for specified camera
    :rtype: np.array
    '''

    data = readCalFile(calfile)
    projMat = data['P%s' % camnum].reshape((3, 4))
    return projMat[:3, :3]

def getPoses(posefile):
    poses = np.empty((0,3,4))
    with open(posefile, 'r') as pf:
        for line in pf.readlines():
            values = line.split()
            values = np.array([float(v) for v in values])
            values = np.reshape(values,(3,4))
            poses = np.append(poses, [values], axis=0)
    return poses