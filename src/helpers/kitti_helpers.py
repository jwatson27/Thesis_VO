# Helper functions for input/output related to the Kitti dataset

import numpy as np
import h5py

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




class IndexConverter():
    def __init__(self, truthFilesList, turnThresh_rad=None):
        self.truthFilesList = truthFilesList
        self.turnThresh_rad = turnThresh_rad
        self.getSeqCounts()
        self.getCumSeqStarts()

    def getSeqCounts(self):
        totalSeqCounts = []
        turnSeqCounts = []
        turnIdxs = []

        for truthFile in self.truthFilesList:
            with h5py.File(truthFile, 'r') as f:
                rot_xyz = np.array(f['rot_xyz'])
                total = len(rot_xyz)
                totalSeqCounts.append(total)

                if self.turnThresh_rad is not None:
                    turningIdxs = np.where(np.abs(rot_xyz[:, 1]) > self.turnThresh_rad)[0]
                    turnIdxs.append(turningIdxs)
                    turn = len(turningIdxs)
                    turnSeqCounts.append(turn)

        totalSeqCounts = np.array(totalSeqCounts)
        self.totalSeqCounts = totalSeqCounts

        if self.turnThresh_rad is not None:
            turnSeqCounts = np.array(turnSeqCounts)
            self.turnSeqCounts = turnSeqCounts
            self.nonTurnSeqCounts = self.totalSeqCounts - self.turnSeqCounts
            self.turnIdxs = turnIdxs


    def getCumSeqStarts(self):
        seqLens = self.totalSeqCounts
        seqStarts = np.insert(seqLens, 0, 0)[:-1]
        cumSeqStarts = np.cumsum(seqStarts)
        self.cumSeqStarts = cumSeqStarts


    def cvtToSeqs(self, indices):
        seqs = [[] for _ in range(len(self.cumSeqStarts))]
        for idx in indices:
            seqNum = np.where(idx>=self.cumSeqStarts)[0][-1]
            idxVal = idx - self.cumSeqStarts[seqNum]
            seqs[seqNum].append(idxVal)
        return seqs

    def cvtToIdxs(self, seqLists):
        idxs = np.empty((0), dtype=np.int)
        for seqNum, seqIdxs in enumerate(seqLists):
            if len(seqIdxs):
                absIdxs = seqIdxs + self.cumSeqStarts[seqNum]
                idxs = np.append(idxs, absIdxs)
        return idxs


