import numpy as np
import glob
import h5py
import os
import sys
from src.helpers.cfg import ThesisConfig

def getCumSeqStarts(seqlens):
    variable = 10
    seqStarts = np.insert(seqlens, 0, 0)[:-1]
    cumSeqStarts = np.cumsum(seqStarts)
    return cumSeqStarts

def cvtToSeqs(indices, seqLens):
    cumSeqStarts = getCumSeqStarts(seqLens)
    seqs = [[] for _ in range(len(cumSeqStarts))]
    for idx in indices:
        seqNum = np.where(idx>=cumSeqStarts)[0][-1]
        idxVal = idx - cumSeqStarts[seqNum]
        seqs[seqNum].append(idxVal)
    return seqs

def cvtSeqIdxsToAbsIdxs(seqIdxs, seqNum):
    cumSeqStarts = getCumSeqStarts(totalSeqCounts)
    absIdxs = seqIdxs+cumSeqStarts[seqNum]
    return absIdxs

def cvtToIdxs(seqList, seqLens):
    cumSeqStarts = getCumSeqStarts(seqLens)
    idxs = np.empty((0), dtype=np.int)
    for i, seqVals in enumerate(seqList):
        if seqVals.tolist():
            absIndexes = cvtSeqIdxsToAbsIdxs(seqVals, i)
            idxs = np.append(idxs, absIndexes)
    return idxs


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
totalSeqCounts = []
turnSeqCounts = []
turnIdxs = []
for seq in usedSeqs:
    truthFile = config.getInputFiles(truthFilesDict, seq)
    if truthFile is None:
        sys.exit()

    with h5py.File(truthFile, 'r') as f:
        rot_xyz = np.array(f['rot_xyz'])
        total = len(rot_xyz)
        totalSeqCounts.append(total)
        turningIdxs = np.where(np.abs(rot_xyz[:,1]) > turnThresh_rad)[0]
        turnIdxs.append(turningIdxs)
        turn = len(turningIdxs)
        turnSeqCounts.append(turn)

totalSeqCounts = np.array(totalSeqCounts)
turnSeqCounts = np.array(turnSeqCounts)
nonTurnSeqCounts = totalSeqCounts - turnSeqCounts



# SPLIT TURNING INDEXES

# Convert seq turning indexes to absolute indexes
turnAbsIdxs = cvtToIdxs(turnIdxs, totalSeqCounts)

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
absIdxs = np.array(range(np.sum(totalSeqCounts)))
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
numTotal = sum(totalSeqCounts)
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
