import os
import numpy as np
import cv2 as cv
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import features as feat

# def destroyWindowsOnKey(delay):
#     if cv.waitKey(delay):
#         cv.destroyAllWindows()



from src.helpers.cfg import ThesisConfig
from src.helpers.kitti_helpers import getCalMat
from src.helpers.coframes import cvtToRpy_sd, cart2sph
import h5py


# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
configFile = 'exp_configs/CNN_test_0.yaml'
config = ThesisConfig(configFile)


# Parameters

kittiSeqs = config.kittiSeqs
usedCams = config.usedCams
normRange = config.thesisKittiParms['normPixelRange']
destImageShape = config.thesisKittiParms['downsampledImageShape']
recalcNormImages = config.expKittiParms['prepared']['runPrep']['recalcNormImages']
showProgress = config.expKittiParms['prepared']['showProg']
numFeat = 0 # TODO: Create parameter in config file
ratio = 0.7 # TODO: Create parameter in config file

# Files
origImageFilesDict = config.kittiOriginal['cams']
calFilesDict       = config.kittiOriginal['cal']
# standardImageFilesDict = config.kittiPrepared['standardImages']



# os.chdir('/home/jwatson27/PycharmProjects/ProjectV3/')

seq = 0
cam = 0

# for cam in usedCams:
#     for seqStr in kittiSeqs:
#         seq = int(seqStr)

# TODO: Get Image Names
origImageNames = config.getInputFiles(origImageFilesDict, seq, cam)
# if (origImageNames is None):
#     continue

# TODO: Get Calibration Files
calFile = config.getInputFiles(calFilesDict, seq)
# if (calFile is None):
#     continue

# TODO: Get Image Size?
image = cv.imread(origImageNames[0])
imageShape = image.shape[:2]

# TODO: Get Camera Calibration Matrix
cameraCalMat = getCalMat(calFile, cam)

# TODO: Get detector and descriptor
SIFT_detector = cv.xfeatures2d.SIFT_create(nfeatures=numFeat)
# SIFT_descriptor = cv.xfeatures2d.SIFT_create(nfeatures=numFeat)





def run_cmp(img_idx, type):

    # TODO: For each image:
    srcImageName = origImageNames[img_idx]
    srcImage = cv.cvtColor(cv.imread(srcImageName), cv.COLOR_BGR2GRAY)
    srcKp, srcDesc = SIFT_detector.detectAndCompute(srcImage, None)

    idx=img_idx+1
    # for idx, origImageName in range(1,len(origImageNames)):
    dstImageName = origImageNames[idx]
    # TODO:  Read it in
    dstImage = cv.cvtColor(cv.imread(dstImageName), cv.COLOR_BGR2GRAY)
    # TODO:  Generate features and descriptors
    dstKp, dstDesc = SIFT_detector.detectAndCompute(dstImage, None)


    # srcKpImage = cv.drawKeypoints(srcImage, srcKp, np.array([]), (0,0,200))
    # dstKpImage = cv.drawKeypoints(dstImage, dstKp, np.array([]), (0,0,200))
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    # ax1.imshow(srcKpImage)
    # ax2.imshow(dstKpImage)


    # TODO: Match Features

    # TODO: Brute Force Matching

    # matches, srcData, dstData = bruteForce(srcImg, dstImg, type, dd, 'data', ratio=rat, numFeat=nf)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)

    # # Cross Check
    # distMethod = cv.NORM_L2
    # bfm = cv.BFMatcher(distMethod, crossCheck=True)
    # matches = bfm.match(srcDesc, dstDesc)
    # matches = sorted(matches, key=lambda x: x.distance)
    # resImg = cv.drawMatches(srcImage, srcKp, dstImage, dstKp, matches, np.array([]))
    # matches = [[m] for m in matches]
    # ax1.imshow(resImg)

    # Ratio Test
    # BFMatcher with default params
    bfm = cv.BFMatcher()
    matches = bfm.knnMatch(srcDesc, dstDesc, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    matches = good
    resImg = cv.drawMatchesKnn(srcImage, srcKp, dstImage, dstKp, matches, np.array([]), flags=2)
    # ax2.imshow(resImg)



    # Task C: Perform Robust Outlier Rejection
    matchedSrcPts = cv.KeyPoint_convert(srcKp, [m[0].queryIdx for m in matches])
    matchedDstPts = cv.KeyPoint_convert(dstKp, [m[0].trainIdx for m in matches])
    essentialMat, mask = cv.findEssentialMat(matchedSrcPts, matchedDstPts, cameraCalMat, cv.FM_RANSAC, 0.999, 1.0)
    # essentialMat, mask = cv.findEssentialMat(src_pts, dst_pts, K, cv.FM_LMEDS, 0.999)
    mask = mask.ravel().tolist()
    mask = [[msk] for msk in mask]

    matchCmp = cv.drawMatchesKnn(srcImage, srcKp, dstImage, dstKp, matches, np.array([]), flags=2,
                                 matchesMask=mask)

    # plt.figure()
    # plt.imshow(matchCmp)

    numMatches = sum([msk[0] for msk in mask])

    print(numMatches)

    # ap = matchedSrcPts
    # bp = matchedDstPts
    # ap_ = np.concatenate((ap, np.ones((len(ap), 1))), axis=1)
    # bp_ = np.concatenate((bp, np.ones((len(bp), 1))), axis=1)
    #
    # invCamCalMat = np.linalg.inv(cameraCalMat)
    # an_ = np.dot(invCamCalMat, ap_.T).T
    # bn_ = np.dot(invCamCalMat, bp_.T).T

    # essMatErr_used = [np.dot(bn_[i].T, np.dot(essentialMat, an_[i])) for i in range(0, len(an_))]
    # essMatErrMean_used = np.mean(np.abs(essMatErr_used))
    # print('Essential Matrix Error (Used Points): {0:.8f}'.format(essMatErrMean_used))

    # TODO: Recover pose

    points, R_ip1_i, t_ip1_ip12i, _ = cv.recoverPose(essentialMat, matchedSrcPts, matchedDstPts, cameraCalMat)
    rot_XYZ = cvtToRpy_sd(np.array([R_ip1_i]))[0,0,:]
    trans_XYZ = t_ip1_ip12i

    # points, R_i_ip1, t_i_i2ip1, _ = cv.recoverPose(essentialMat, matchedSrcPts, matchedDstPts, cameraCalMat)
    # rot_XYZ = cvtToRpy_sd(np.array([R_i_ip1]))[0,0,:]
    # trans_XYZ = t_i_i2ip1


    # print(trans_XYZ, trans_cmp)


    truthFilesDict = config.kittiPrepared['truth']
    truthFile = config.getInputFiles(truthFilesDict, seq)
    with h5py.File(truthFile, 'r') as f:
        rot_xyz = np.array(f['rot_xyz'])
        trans_xyz = np.array(f['trans_xyz'])
        trans_rtp = np.array(f['trans_rtp'])




    if type=='rot':
        rot_XYZ = rot_XYZ * 180 / np.pi
        rot_xyz = rot_xyz[img_idx] * 180 / np.pi
        print('Epi Rot  [%d]: [%12f, %12f, %12f]' % (img_idx, rot_XYZ[0], rot_XYZ[1], rot_XYZ[2]))
        print('True Rot [%d]: [%12f, %12f, %12f]' % (img_idx, rot_xyz[0], rot_xyz[1], rot_xyz[2]))
    else:
        trans_xyz = trans_xyz[img_idx]
        trans_xyz = trans_xyz/np.sqrt(np.sum(trans_xyz**2))
        print('Epi Trans  [%d]: [%12f, %12f, %12f]' % (img_idx, trans_XYZ[0], trans_XYZ[1], trans_XYZ[2]))
        print('True Trans [%d]: [%12f, %12f, %12f]' % (img_idx, trans_xyz[0], trans_xyz[1], trans_xyz[2]))


for i in range(1100,1200):
    run_cmp(i, 'trans')
# run_cmp(0)
# run_cmp(100)
# run_cmp(267)
# run_cmp(1121)
# run_cmp(1122)
# run_cmp(1123)
# run_cmp(1124)
# run_cmp(1125)
# run_cmp(1126)




# Prep next iteration
# srcKp, srcDesc = dstKp, dstDesc










# def detectors(det='sift', nFeat=0):
#     if det == 'fast':
#         return (cv.FastFeatureDetector_create(), 'normal')
#     elif det == 'orb':
#         if nFeat==0:
#             return (cv.ORB_create(), 'binary')
#         else:
#             return (cv.ORB_create(nfeatures=nFeat), 'binary')
#     elif det == 'sift':
#         if nFeat == 0:
#             return (cv.xfeatures2d.SIFT_create(), 'normal')
#         else:
#             return (cv.xfeatures2d.SIFT_create(nfeatures=nFeat), 'normal')
#
#
#
#
# def descriptors(desc='sift', nFeat=0):
#     if desc == 'brief':
#         return (cv.xfeatures2d.BriefDescriptorExtractor_create(), 'binary')
#     elif desc == 'orb':
#         if nFeat == 0:
#             return (cv.ORB_create(), 'binary')
#         else:
#             return (cv.ORB_create(nfeatures=nFeat), 'binary')
#     elif desc == 'sift':
#         if nFeat == 0:
#             return (cv.xfeatures2d.SIFT_create(), 'normal')
#         else:
#             return (cv.xfeatures2d.SIFT_create(nfeatures=nFeat), 'normal')
#
#
#
#
#
# def bruteForce(im0, im1, type='', detDesc='sift', output='data', ratio=0.7, numFeat=0):
#
#     if isinstance(detDesc,str):
#         # using same method for detector/descriptor
#         featDet, descType = detectors(detDesc, numFeat)
#         # featDesc, descType = descriptors(detDesc, numFeat)
#
#         kp0, des0 = featDet.detectAndCompute(im0, None)
#         kp1, des1 = featDet.detectAndCompute(im1, None)
#
#     else:
#         featDet, _ = detectors(detDesc[0], numFeat)
#         featDesc, descType = descriptors(detDesc[1], numFeat)
#
#         kp0 = featDet.detect(im0, None)
#         kp1 = featDet.detect(im1, None)
#
#         kp0, des0 = featDesc.compute(im0, kp0)
#         kp1, des1 = featDesc.compute(im1, kp1)
#
#     if type=='cross':
#         # BFMatcher with Cross Check
#         if descType == 'binary':
#             distType = 'Hamming'
#             distMethod = cv.NORM_HAMMING
#         else:
#             distType = 'L2 Norm'
#             distMethod = cv.NORM_L2
#
#         bfm = cv.BFMatcher(distMethod, crossCheck=True)
#         matches = bfm.match(des0, des1)
#         matches = sorted(matches, key=lambda x: x.distance)
#
#         title = '%s - matches: %s' % (distType, len(matches))
#
#         if output == 'figure' or output == 'figdata':
#             resImg = cv.drawMatches(im0, kp0, im1, kp1, matches, np.array([]))
#
#         matches = [[m] for m in matches]
#
#     else:
#         # BFMatcher with default params
#         bfm = cv.BFMatcher()
#         matches = bfm.knnMatch(des0, des1, k=2)
#         title = 'Default - matches: %s' % len(matches)
#
#         if type=='ratio':
#             # Apply ratio test
#             good = []
#             for m, n in matches:
#                 if m.distance < ratio * n.distance:
#                     good.append([m])
#             matches = good
#             title = 'Ratio - matches: %s, ratio: %s' % (len(matches), ratio)
#
#         if output == 'figure' or output == 'figdata':
#             resImg = cv.drawMatchesKnn(im0, kp0, im1, kp1, matches, np.array([]), flags=2)
#
#     if output == 'figure' or output == 'figdata':
#         plt.figure()
#         plt.title(title)
#         plt.imshow(resImg)
#         plt.show()
#
#     if output == 'data' or output == 'figdata':
#         img0_data = {'kp': kp0, 'des': des0}
#         img1_data = {'kp': kp1, 'des': des1}
#         return (matches, img0_data, img1_data)
#
#
# def maxCompare(srcImg, dstImg, K, type='ratio', dd='sift', mthd='ransac', output='data', nf=0, rat=0.7):
#     matches, srcData, dstData = bruteForce(srcImg, dstImg, type, dd, 'data', ratio=rat, numFeat=nf)
#
#     if isinstance(dd,str):
#         det=dd
#         desc=dd
#     else:
#         det=dd[0]
#         desc=dd[1]
#
#     # Task C: Perform Robust Outlier Rejection
#     src_pts = cv.KeyPoint_convert(srcData['kp'], [m[0].queryIdx for m in matches])
#     dst_pts = cv.KeyPoint_convert(dstData['kp'], [m[0].trainIdx for m in matches])
#
#     if mthd == 'ransac':
#         essentialMat, mask = cv.findEssentialMat(src_pts, dst_pts, K, cv.FM_RANSAC, 0.999, 1.0)
#     elif mthd == 'lmeds':
#         essentialMat, mask = cv.findEssentialMat(src_pts, dst_pts, K, cv.FM_LMEDS, 0.999)
#     mask = mask.ravel().tolist()
#     mask = [[msk] for msk in mask]
#
#     matchCmp = cv.drawMatchesKnn(srcImg, srcData['kp'], dstImg, dstData['kp'], matches, np.array([]), flags=2, matchesMask=mask)
#
#     finalMatches = sum([msk[0] for msk in mask])
#
#     print (finalMatches)
#
#     if output=='fig' or output=='figure' or output=='figdata':
#         plt.figure()
#         plt.imshow(matchCmp)
#         if type=='':
#             type = 'default'
#         title = ('%s, %s, %s/%s - matches: %s' % (type, mthd, det, desc, finalMatches))
#         if type=='ratio':
#             title = ('%s, ratio: %s' % (title, str(rat)))
#         plt.title(title)
#
#     if output=='data' or output=='figdata':
#         return (essentialMat, src_pts, dst_pts, matches, mask)