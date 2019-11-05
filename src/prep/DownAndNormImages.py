import cv2 as cv
import numpy as np
import os
import sys
import h5py

from src.helpers.cfg import ThesisConfig

# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
config = ThesisConfig(configFile)

# Parameters
kittiSeqs = config.kittiSeqs
usedCams = config.usedCams
normRange = config.thesisKittiParms['normPixelRange']
destImageShape = config.thesisKittiParms['downsampledImageShape']
recalcNormImages = config.expKittiParms['prepared']['runPrep']['recalcNormImages']
showProgress = config.expKittiParms['prepared']['showProg']

# Files
standardImageFilesDict = config.kittiPrepared['standardImages']
normImageFilesDict = config.kittiNormalized['normImages']





print()
print('v Downsampling Images and Normalizing Pixels')

for cam in usedCams:
    for seqStr in kittiSeqs:
        seq = int(seqStr)
        print('%sReading Camera %s, Sequence %02d' % (' ' * 2, cam, seq))

        standardImages = config.getInputFiles(standardImageFilesDict, seq, cam)
        if (standardImages is None):
            continue

        normImages = config.getOutputFiles(normImageFilesDict, recalcNormImages, seq, cam)
        if (normImages is None):
            continue

        normImageFolder = config.getFolderRef(normImages)

        numImages = len(standardImages)
        for idx, standardImageName in enumerate(standardImages):

            if showProgress:
                percentComplete = int(idx/numImages*100)
                if divmod(idx,300)[1]==0:
                    print('Percent Complete: %d%%' % percentComplete)

            srcImage = cv.cvtColor(cv.imread(standardImageName), cv.COLOR_BGR2GRAY)
            dstImage = cv.resize(srcImage, dsize=destImageShape[::-1], interpolation=cv.INTER_AREA)

            srcImage = dstImage
            dstImage = cv.normalize(srcImage, None, normRange[0], normRange[1], cv.NORM_MINMAX, dtype=cv.CV_32F)


            normFile = os.path.basename(standardImageName).split('.')[0] + normImageFilesDict['type']
            saveFile = os.path.join(normImageFolder, normFile)

            with h5py.File(saveFile, 'w') as f:
                os.chmod(saveFile, 0o666)
                f.create_dataset('image', data=dstImage)


print('^ Image Downsampling and Normalizing Complete')
print()