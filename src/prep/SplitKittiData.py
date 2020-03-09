import numpy as np
import glob
import h5py
import os
import sys
from src.helpers.cfg import ThesisConfig
from src.helpers.kitti_helpers import IndexConverter


# TODO: Handle multiple cameras

# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)

# Parameters
kittiSeqs = config.kittiSeqs
usedCams = config.usedCams
usedSeqs = config.usedSeqs
splitFracs = config.splitFracs
turnThresh_rad = config.thesisKittiParms['turnThreshold'] * np.pi/180

# Files
truthFilesDict = config.kittiPrepared['truth']
splitFilesDict = config.kittiPrepared['split']



print()
print('v Splitting Dataset')

splitFile = config.getOutputFiles(splitFilesDict)
if splitFile is None:
    print('^ Dataset Splitting Complete')
    print()
    sys.exit()

# Check if file exists
# Get Sequences Counts and Turning Indexes
truthFilesList = []
for seq in usedSeqs:
    truthFilesList.append(config.getInputFiles(truthFilesDict, seq))


idxCvt = IndexConverter(truthFilesList, turnThresh_rad)


# SPLIT TURNING INDEXES

# Convert seq turning indexes to absolute indexes
turnAbsIdxs = idxCvt.cvtToIdxs(idxCvt.turnIdxs)

# Determine total number of image pair indexes for turning
numTurn = len(turnAbsIdxs)

# Determine number of image pairs for train, validation, and test for turning
numTrainTurn, numValTurn, numTestTurn = np.array(np.round(numTurn*splitFracs), dtype=np.int)
numTrainTurn += numTurn-(numTrainTurn + numValTurn + numTestTurn)

# Shuffle turning indexes
origRandomTurnIdxs = np.random.choice(turnAbsIdxs, len(turnAbsIdxs), replace=False)
randTurnIdxs = origRandomTurnIdxs

# Randomly sample the turning indexes to get the turning test indexes
testTurnAbsIdxs = randTurnIdxs[:numTestTurn]
randTurnIdxs = np.delete(randTurnIdxs, range(numTestTurn))

# Randomly sample the remaining turning indexes to get the turning training indexes
trainTurnAbsIdxs = randTurnIdxs[:numTrainTurn]

# The remaining turning indexes are validation indexes
valTurnAbsIdxs = randTurnIdxs[numTrainTurn:]



# SPLIT NON-TURNING INDEXES

# Convert seq non-turning indexes to absolute indexes
absIdxs = np.array(range(np.sum(idxCvt.totalSeqCounts)))
nonTurnAbsIdxs = np.setdiff1d(absIdxs, turnAbsIdxs)

# Determine total number of image pair indexes for non-turning
numNonTurn = len(nonTurnAbsIdxs)

# Determine number of image pairs for train, validation, and test for non-turning
numTrainNonTurn, numValNonTurn, numTestNonTurn = np.array(np.round(numNonTurn*splitFracs), dtype=np.int)
numTrainNonTurn += numNonTurn-(numTrainNonTurn + numValNonTurn + numTestNonTurn)

# Shuffle non-turning indexes
origRandomNonTurnIdxs = np.random.choice(nonTurnAbsIdxs, len(nonTurnAbsIdxs), replace=False)
randNonTurnIdxs = origRandomNonTurnIdxs

# Randomly sample the non-turning indexes in the other sequences to get the rest of the test indexes
testNonTurnAbsIdxs = randNonTurnIdxs[:numTestNonTurn]
randNonTurnIdxs = np.delete(randNonTurnIdxs, range(numTestNonTurn))

# Randomly sample the remaining non-turning indexes to get the non-turning training indexes
trainNonTurnAbsIdxs = randNonTurnIdxs[:numTrainNonTurn]

# The remaining non-turning indexes are validation indexes
valNonTurnAbsIdxs = randNonTurnIdxs[numTrainNonTurn:]



# Save Turning/Non-Turning to split files

with h5py.File(splitFile, 'w') as f:
    os.chmod(splitFile, 0o666)
    f.create_dataset('trainTurnIdxs', data=trainTurnAbsIdxs)
    f.create_dataset('valTurnIdxs', data=valTurnAbsIdxs)
    f.create_dataset('testTurnIdxs', data=testTurnAbsIdxs)
    f.create_dataset('trainNonTurnIdxs', data=trainNonTurnAbsIdxs)
    f.create_dataset('valNonTurnIdxs', data=valNonTurnAbsIdxs)
    f.create_dataset('testNonTurnIdxs', data=testNonTurnAbsIdxs)



# Check Values - DEBUGGING
numTotal = sum(idxCvt.totalSeqCounts)
print('Turning')
print(numTrainTurn, numValTurn, numTestTurn)
print(len(trainTurnAbsIdxs), len(valTurnAbsIdxs), len(testTurnAbsIdxs))

print('Not Turning')
print(numTrainNonTurn, numValNonTurn, numTestNonTurn)
print(len(trainNonTurnAbsIdxs), len(valNonTurnAbsIdxs), len(testNonTurnAbsIdxs))

print('Total')
print(len(trainTurnAbsIdxs)+len(trainNonTurnAbsIdxs),
      len(valTurnAbsIdxs)+len(valNonTurnAbsIdxs),
      len(testTurnAbsIdxs)+len(testNonTurnAbsIdxs))
print(numTotal*splitFracs)

print(len(trainTurnAbsIdxs)+len(trainNonTurnAbsIdxs)+
      len(valTurnAbsIdxs)+len(valNonTurnAbsIdxs)+
      len(testTurnAbsIdxs)+len(testNonTurnAbsIdxs))
print(numTotal)


print('^ Dataset Splitting Complete')
print()
