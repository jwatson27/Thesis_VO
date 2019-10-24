import numpy as np
import cv2 as cv
import glob
import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.helpers.kitti_helpers import getCalMat
from src.helpers.cfg import ThesisConfig



# Helper Functions

def getCornerStandard(shape, ccm):
    height, width = shape
    # Convert pixel coordinates to normalized coordinates
    cornerPix_homg = np.array([[0, width - 1, width - 1, 0],
                               [0, 0, height - 1, height - 1],
                               np.ones((4,))])
    cornerStandard = np.dot(np.linalg.inv(ccm), cornerPix_homg)
    return cornerStandard[:-1] / cornerStandard[-1]


def plotRect(corners, fmt='', color=None):
    if corners.shape[0] > corners.shape[1]:
        corners = corners.T
    rectPts = np.append(corners, corners[:,0:1], axis=1)
    plt.plot(rectPts[0,:], rectPts[1,:], fmt, c=color)


def getMaps(destShape, corners, ccm):
    height, width = destShape

    # get normalized overlay grid of destination size
    spreadX = np.linspace(corners[0,0], corners[0,1], width)
    spreadY = np.linspace(corners[1,1], corners[1,2], height)
    gridX, gridY = np.meshgrid(spreadX, spreadY)
    lin_homg_norm = np.array([gridX.ravel(), gridY.ravel(), np.ones_like(gridX).ravel()])

    # get map for sequence
    mapPts = np.dot(ccm, lin_homg_norm) # convert to pixel coordinates
    mapX, mapY = mapPts[:-1]/mapPts[-1]  # ensure homogeneity
    mapX = mapX.reshape(height, width).astype(np.float32)
    mapY = mapY.reshape(height, width).astype(np.float32)

    return (mapX, mapY)


# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)

# Parameters
kittiSeqs        = config.kittiSeqs
usedCams         = config.usedCams
destImageShape   = config.thesisKittiParms['standardImageShape']
recalcStandardImages = config.expKittiParms['prepared']['recalcStandardImages']
showFigures      = config.expKittiParms['prepared']['showPlots']
showProgress     = config.expKittiParms['prepared']['showProg']

# Files
origImageFilesDict = config.kittiOriginal['cams']
calFilesDict       = config.kittiOriginal['cal']
standardImageFilesDict = config.kittiPrepared['standardImages']


# Get normalized corner points for each sequence
standardDict = {}
for cam in usedCams:
    for seqStr in kittiSeqs:
        seq = int(seqStr)

        origImageNames = config.getInputFiles(origImageFilesDict, seq, cam)
        if (origImageNames is None):
            continue

        calFile = config.getInputFiles(calFilesDict, seq)
        if (calFile is None):
            continue

        image = cv.imread(origImageNames[0])
        imageShape = image.shape[:2]

        cameraCalMat = getCalMat(calFile, cam)
        standardDict[seqStr] = getCornerStandard(imageShape, cameraCalMat)

# TODO: SAVE NORM DICT NOT FIGURE
# TODO: GENERATE FIGURE IN DIFFERENT FILE

# if showFigures:
#     # Plot standardized rectangles on figure
#     lgnd = []
#     fig = plt.figure()
#     for seqStr in standardDict:
#         plotRect(standardDict[seqStr])
#         lgnd.append(seqStr)
#     plt.legend(lgnd)

# Select overlap region
xleft  = str(kittiSeqs[np.argmax([standardDict[key][0,0] for key in standardDict])])
xright = str(kittiSeqs[np.argmin([standardDict[key][0,1] for key in standardDict])])
ylower = str(kittiSeqs[np.argmax([standardDict[key][1,1] for key in standardDict])])
yupper = str(kittiSeqs[np.argmin([standardDict[key][1,2] for key in standardDict])])

standardOverlap = np.array([[standardDict[xleft][0,0],  standardDict[xright][0,1], standardDict[xright][0,2], standardDict[xleft][0,3]],
                            [standardDict[ylower][1,0], standardDict[ylower][1,1], standardDict[yupper][1,2], standardDict[yupper][1,3]]])
standardDict['overlap'] = standardOverlap



# TODO: SAVE NORM DICT NOT FIGURE
# TODO: GENERATE FIGURE IN DIFFERENT FILE
#
# if showFigures:
#     # Plot Overlap
#     fig = plt.gcf()
#     plotRect(standardDict['overlap'], '--', 'k')
#     plt.xlabel('Standardized X-direction')
#     plt.ylabel('Standardized Y-direction')
#     lgnd.append('overlap')
#     plt.legend(lgnd)
#     os.getcwd()
#     os.listdir()
#     figPath = config.thesis['resultPaths']['figures']['dir']
#     if not os.path.isdir(figPath):
#         os.makedirs(figPath)
#     plt.savefig(os.path.join(figPath, 'image overlap.png'))


# Remap Images to Overlap Region
print()
print('v Standardizing Images')
for cam in usedCams:
    for seqStr in kittiSeqs:
        seq = int(seqStr)
        print('%sReading Camera %s, Sequence %02d' % (' '*2, cam, seq))

        calFile = config.getInputFiles(calFilesDict, seq)
        if (calFile is None):
            continue

        origImages = config.getInputFiles(origImageFilesDict, seq, cam)
        if (origImages is None):
            continue

        standardImages = config.getOutputFiles(standardImageFilesDict, recalcStandardImages, seq, cam)
        if (standardImages is None):
            continue

        standardImageFolder = config.getFolderRef(standardImages)

        # get cal matrix for sequence
        cameraCalMat = getCalMat(calFile, cam)
        map_x, map_y = getMaps(destImageShape, standardDict['overlap'], cameraCalMat)

        numImages = len(origImages)
        for idx, origImageName in enumerate(origImages):

            if showProgress:
                percentComplete = int(idx/numImages*100)
                if divmod(idx,300)[1]==0:
                    print('Percent Complete: %d%%' % percentComplete)

            srcImage = cv.cvtColor(cv.imread(origImageName), cv.COLOR_BGR2GRAY)
            dstImage = cv.remap(srcImage, map_x, map_y, cv.INTER_LINEAR)

            saveName = os.path.join(standardImageFolder, os.path.basename(origImageName))
            cv.imwrite(saveName, dstImage)

print('^ Image Standardization Complete')
print()