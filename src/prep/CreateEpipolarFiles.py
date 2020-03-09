import os
import numpy as np
import cv2 as cv
import sys
import h5py

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from src.helpers.cfg import ThesisConfig
from src.helpers.kitti_helpers import getCalMat
from src.helpers.coframes import cvtToRpy_sd, cart2sph
from src.helpers.helper_functions import getKpsAndDescs, \
    matchAndRatioTest, compareTruthToEpipolar, outlierRejectionRANSAC


# compareTruthToEpipolar('exp_configs/scale_test_3.yaml',0,0,range(100),'rot')


# Load Configuration
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]
configFile = 'exp_configs/scale_test_3.yaml'
config = ThesisConfig(configFile)


# Parameters

kittiSeqs = config.kittiSeqs
usedCams = config.usedCams
normRange = config.thesisKittiParms['normPixelRange']
destImageShape = config.thesisKittiParms['downsampledImageShape']
recalcEpiData = config.expKittiParms['prepared']['runPrep']['recalcEpi']
showProgress = config.expKittiParms['prepared']['showProg']
numFeat = config.thesisKittiParms['epiNumFeatures']
epiRatio = config.thesisKittiParms['epiRatio']

# Files
origImageFilesDict = config.kittiOriginal['cams']
calFilesDict       = config.kittiOriginal['cal']
epiFilesDict = config.kittiPrepared['epipolar']


recalcEpiData = True
# kittiSeqs = np.delete(kittiSeqs, 1)
numFeat = 1000


# Get Sift Detector
SIFT_detector = cv.xfeatures2d.SIFT_create(nfeatures=numFeat)

print()
print('v Creating Epipolar Data')
for cam in usedCams:
    for seqStr in kittiSeqs:
        seq = int(seqStr)
        print('%sReading Camera %s, Sequence %02d' % (' ' * 2, cam, seq))

        # Get image names
        origImageNames = config.getInputFiles(origImageFilesDict, seq, cam)
        # if (origImageNames is None):
        #     continue

        # get calibration files
        calFile = config.getInputFiles(calFilesDict, seq)
        # if (calFile is None):
        #     continue

        epiFile = config.getOutputFiles(epiFilesDict, recalcEpiData, seq, cam)
        # if epiFile is None:
        #     continue

        # get calibration matrix
        cameraCalMat = getCalMat(calFile, cam)

        epi_rot_xyz = np.empty((0, 3))
        epi_trans_xyz = np.empty((0, 3))

        numImagePairs = len(origImageNames)-1

        # get first image keypoints and descriptors
        srcKp, srcDesc = getKpsAndDescs(SIFT_detector, origImageNames[0])

        for idx, origImageName in enumerate(origImageNames[1:]):
            if showProgress:
                percentComplete = int(idx/numImagePairs*100)
                if divmod(idx,100)[1]==0:
                    print('Percent Complete: %d%%' % percentComplete)

            # srcKp, srcDesc = getKpsAndDescs(SIFT_detector, origImageNames[pairIdx])
            # origImageName = origImageNames[pairIdx+1]

            dstKp, dstDesc = getKpsAndDescs(SIFT_detector, origImageName)

            matches = matchAndRatioTest(srcDesc, dstDesc, epiRatio)

            # Match comparison image before RANSAC
            # srcImage = cv.imread(origImageNames[pairIdx])
            # dstImage = cv.imread(origImageNames[pairIdx+1])
            # resImg = cv.drawMatchesKnn(srcImage, srcKp, dstImage, dstKp, matches, np.array([]), flags=2)
            # import matplotlib.pyplot as plt
            # plt.imshow(resImg)

            matchedSrcPts = cv.KeyPoint_convert(srcKp, [m[0].queryIdx for m in matches])
            matchedDstPts = cv.KeyPoint_convert(dstKp, [m[0].trainIdx for m in matches])

            # outlier rejection with RANSAC
            essentialMat, mask = outlierRejectionRANSAC(matchedSrcPts, matchedDstPts, cameraCalMat)

            numMatches = sum([msk[0] for msk in mask])
            print(numMatches)

            # Apply RANSAC mask
            # bool_mask = np.array(mask, dtype=bool).T[0]
            # maskedMatchedSrcPts = matchedSrcPts[bool_mask]
            # maskedMatchedDstPts = matchedDstPts[bool_mask]

            # Match comparison image after RANSAC
            # matchCmp = cv.drawMatchesKnn(srcImage, srcKp, dstImage, dstKp, np.array(matches)[bool_mask], np.array([]), flags=2) #, matchesMask=mask)
            # plt.imshow(matchCmp)

            # recover pose
            points, R_ip1_i, t_ip1_ip12i, _ = cv.recoverPose(essentialMat, matchedSrcPts, matchedDstPts, cameraCalMat)
            rot_ip1_i = cvtToRpy_sd(np.array([R_ip1_i]))[0, 0, :]
            trans_ip1_ip12i = t_ip1_ip12i[:, 0]

            # recover pose
            # points, R_ip1_i, t_ip1_ip12i, _ = cv.recoverPose(essentialMat, maskedMatchedSrcPts, maskedMatchedDstPts, cameraCalMat)
            # rot_ip1_i = cvtToRpy_sd(np.array([R_ip1_i]))[0,0,:]
            # trans_ip1_ip12i = t_ip1_ip12i[:,0]

            # append to array
            epi_rot_xyz = np.append(epi_rot_xyz, [rot_ip1_i], axis=0)
            epi_trans_xyz = np.append(epi_trans_xyz, [trans_ip1_ip12i], axis=0)

            # prep next iteration
            srcKp, srcDesc = dstKp, dstDesc

        # Get two angles version of trans data
        epi_trans_rtp = cart2sph(epi_trans_xyz)


        epiFileMasked = '.'.join((epiFile.rsplit('.')[0] + '_1000', epiFile.rsplit('.')[1]))
        # Save to File
        with h5py.File(epiFileMasked, 'w') as f:
            f.create_dataset('epi_rot_xyz', data=epi_rot_xyz)
            f.create_dataset('epi_trans_xyz', data=epi_trans_xyz)
            f.create_dataset('epi_trans_rtp', data=epi_trans_rtp)

        print('^ Epipolar Data Created')
        print()



# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
#
# truthFile = '/clean_dataset/kitti_odom/01/truth.hdf5'
# with h5py.File(truthFile, 'r') as f:
#     true_rot_xyz = np.array(f['rot_xyz'])
#     true_trans_rtp = np.array(f['trans_rtp'])
#     true_trans_xyz = np.array(f['trans_xyz'])
#
# epiFile = '/clean_dataset/kitti_odom/01/cam_0/epipolar.hdf5'
# with h5py.File(epiFile, 'r') as f:
#     epi_rot_xyz = np.array(f['epi_rot_xyz'])
#     epi_trans_xyz = np.array(f['epi_trans_xyz'])
#     # epi_trans_rtp = np.array(f['epi_trans_rtp'])
#
#
# epiFileMasked = '/clean_dataset/kitti_odom/01/cam_0/epipolar_masked.hdf5'
# with h5py.File(epiFileMasked, 'r') as f:
#     epim_rot_xyz = np.array(f['epi_rot_xyz'])
#     epim_trans_xyz = np.array(f['epi_trans_xyz'])
#     epim_trans_rtp = np.array(f['epi_trans_rtp'])
#
#
# epiFileMasked1000 = '/clean_dataset/kitti_odom/01/cam_0/epipolar_masked1000.hdf5'
# with h5py.File(epiFileMasked1000, 'r') as f:
#     epim1000_rot_xyz = np.array(f['epi_rot_xyz'])
#     epim1000_trans_xyz = np.array(f['epi_trans_xyz'])
#     epim1000_trans_rtp = np.array(f['epi_trans_rtp'])
#
#
# plt.figure()
# plt.scatter(true_rot_xyz[:,0], epi_rot_xyz[:,0])
# plt.figure()
# plt.scatter(true_rot_xyz[:,0], epim_rot_xyz[:,0])
# plt.figure()
# plt.scatter(true_rot_xyz[:,0], epim1000_rot_xyz[:,0])





# truthFile = '/clean_dataset/kitti_odom/00/truth.hdf5'
# with h5py.File(truthFile, 'r') as f:
#     true_trans_rtp = np.array(f['trans_rtp'])
#     true_trans_xyz = np.array(f['trans_xyz'])
#
#
#
#
#
#
#
# import matplotlib.pyplot as plt
# from src.helpers.cfg import ThesisConfig
# import h5py
# import numpy as np
#
# configFile = None
# if len(sys.argv)>1:
#     configFile = sys.argv[1]
# configFile = 'exp_configs/scale_test_3.yaml'
# config = ThesisConfig(configFile)
#
# epiFilesDict = config.kittiPrepared['epipolar']
# truthFilesDict = config.kittiPrepared['truth']
#
# seq = 0
# cam = 0
#
#
# epiFile = config.getInputFiles(epiFilesDict, seq, cam)
# with h5py.File(epiFile, 'r') as f:
#     epi_rot_xyz = np.array(f['epi_rot_xyz'])
#     epi_trans_xyz = np.array(f['epi_trans_xyz'])
# epi_xyz = np.concatenate((epi_rot_xyz, epi_trans_xyz),axis=1)
#
# epiFileMasked = '.'.join((epiFile.rsplit('.')[0] + '_masked', epiFile.rsplit('.')[1]))
# with h5py.File(epiFileMasked, 'r') as f:
#     epi_rot_xyz_masked = np.array(f['epi_rot_xyz'])
#     epi_trans_xyz_masked = np.array(f['epi_trans_xyz'])
# epi_xyz_masked = np.concatenate((epi_rot_xyz_masked, epi_trans_xyz_masked),axis=1)
#
# truthFile = config.getInputFiles(truthFilesDict, seq)
# with h5py.File(truthFile, 'r') as f:
#     true_rot_xyz = np.array(f['rot_xyz'])
#     true_trans_xyz = np.array(f['trans_xyz'])
# true_trans_xyz_norm = true_trans_xyz/np.array([np.sqrt(np.sum(true_trans_xyz**2,axis=1))]).T
# true_xyz = np.concatenate((true_rot_xyz, true_trans_xyz_norm),axis=1)
#
#
#
#
#
# # # Plot epipolar data vs. truth
# # lines = []
# # for i in range(y_true_real.shape[1]):
# #     truth = y_true_real[:,i]
# #     preds = y_pred_real[:,i]
# #     min_val = np.min((truth,preds))
# #     max_val = np.max((truth,preds))
# #     min_dec_places = -int(np.round(np.log10(np.abs(min_val))))
# #     max_dec_places = -int(np.round(np.log10(np.abs(max_val))))
# #     if min_dec_places<0:
# #         min_dec_places = 0
# #     if max_dec_places<0:
# #         max_dec_places = 0
# #     line_min = np.round(min_val-0.5*10**-min_dec_places,decimals=min_dec_places)
# #     line_max = np.round(max_val+0.5*10**-max_dec_places,decimals=max_dec_places)
# #     line = np.array([line_min, line_max])
# #     lines.append(line)
#
# numOutputs = true_xyz.shape[1]
# name = 'scale_test_3'
#
# title = 'True vs. Epipolar'
# rot_types = ['Rotation delta_X',
#              'Rotation delta_Y',
#              'Rotation delta_Z']
# trans_types = ['Translation X',
#                'Translation Y',
#                'Translation Z']
# scale_types = ['Translation Scale']
# rot_units = 'rad'
# trans_units = 'meters'
#
# title_types = np.empty(0)
# units = np.empty(0)
# if numOutputs==1:
#     title_types = scale_types
#     units = [trans_units]
# elif numOutputs==3:
#     title_types = trans_types
#     units = np.append(units, [trans_units]*3)
# else:
#     title_types = np.append(title_types, rot_types)
#     title_types = np.append(title_types, trans_types)
#     units = np.append(units, [rot_units]*3)
#     units = np.append(units, [trans_units]*3)
#
# lines = []
# for i in range(numOutputs):
#     truth = true_xyz[:,i]
#     preds = epi_xyz[:,i]
#     min_val = np.min((truth,preds))
#     max_val = np.max((truth,preds))
#     min_dec_places = -int(np.round(np.log10(np.abs(min_val))))
#     max_dec_places = -int(np.round(np.log10(np.abs(max_val))))
#     if min_dec_places<0:
#         min_dec_places = 0
#     if max_dec_places<0:
#         max_dec_places = 0
#     line_min = np.round(min_val-0.5*10**-min_dec_places,decimals=min_dec_places)
#     line_max = np.round(max_val+0.5*10**-max_dec_places,decimals=max_dec_places)
#     line = np.array([line_min, line_max])
#     lines.append(line)
#
#
#
#
#
# for i in range(numOutputs):
#     plt.figure()
#     plt.scatter(true_xyz[:,i], epi_xyz[:,i])
#     # plt.plot(lines[i], lines[i], c='r')
#     plt.title('%s -- %s %s' % (name, title, title_types[i]))
#     plt.xlabel('True (%s)' % units[i])
#     plt.ylabel('Predicted (%s)' % units[i])
#     plt.axis('equal')
#     # saveFile = '%s.png' % os.path.join(figFolder,title_types[i])
#     # plt.savefig(saveFile)
# # plt.show(block=True)
#
#
#
#
# title = 'True vs. Epipolar Masked'
#
# lines = []
# for i in range(numOutputs):
#     truth = true_xyz[:,i]
#     preds = epi_xyz_masked[:,i]
#     min_val = np.min((truth,preds))
#     max_val = np.max((truth,preds))
#     min_dec_places = -int(np.round(np.log10(np.abs(min_val))))
#     max_dec_places = -int(np.round(np.log10(np.abs(max_val))))
#     if min_dec_places<0:
#         min_dec_places = 0
#     if max_dec_places<0:
#         max_dec_places = 0
#     line_min = np.round(min_val-0.5*10**-min_dec_places,decimals=min_dec_places)
#     line_max = np.round(max_val+0.5*10**-max_dec_places,decimals=max_dec_places)
#     line = np.array([line_min, line_max])
#     lines.append(line)
#
# for i in range(numOutputs):
#     plt.figure()
#     plt.scatter(true_xyz[:,i], epi_xyz_masked[:,i])
#     # plt.plot(lines[i], lines[i], c='r')
#     plt.title('%s -- %s %s' % (name, title, title_types[i]))
#     plt.xlabel('True (%s)' % units[i])
#     plt.ylabel('Predicted (%s)' % units[i])
#     plt.axis('equal')
#     # saveFile = '%s.png' % os.path.join(figFolder,title_types[i])
#     # plt.savefig(saveFile)
# # plt.show(block=True)
#
#
#
#
#
#
#     # # Cross Check
#     # distMethod = cv.NORM_L2
#     # bfm = cv.BFMatcher(distMethod, crossCheck=True)
#     # matches = bfm.match(srcDesc, dstDesc)
#     # matches = sorted(matches, key=lambda x: x.distance)
#     # resImg = cv.drawMatches(srcImage, srcKp, dstImage, dstKp, matches, np.array([]))
#     # matches = [[m] for m in matches]
#     # ax1.imshow(resImg)
#
#
#
#
# # def detectors(det='sift', nFeat=0):
# #     if det == 'fast':
# #         return (cv.FastFeatureDetector_create(), 'normal')
# #     elif det == 'orb':
# #         if nFeat==0:
# #             return (cv.ORB_create(), 'binary')
# #         else:
# #             return (cv.ORB_create(nfeatures=nFeat), 'binary')
# #     elif det == 'sift':
# #         if nFeat == 0:
# #             return (cv.xfeatures2d.SIFT_create(), 'normal')
# #         else:
# #             return (cv.xfeatures2d.SIFT_create(nfeatures=nFeat), 'normal')
# #
# #
# #
# #
# # def descriptors(desc='sift', nFeat=0):
# #     if desc == 'brief':
# #         return (cv.xfeatures2d.BriefDescriptorExtractor_create(), 'binary')
# #     elif desc == 'orb':
# #         if nFeat == 0:
# #             return (cv.ORB_create(), 'binary')
# #         else:
# #             return (cv.ORB_create(nfeatures=nFeat), 'binary')
# #     elif desc == 'sift':
# #         if nFeat == 0:
# #             return (cv.xfeatures2d.SIFT_create(), 'normal')
# #         else:
# #             return (cv.xfeatures2d.SIFT_create(nfeatures=nFeat), 'normal')
# #
# #
# #
# #
# #
# # def bruteForce(im0, im1, type='', detDesc='sift', output='data', ratio=0.7, numFeat=0):
# #
# #     if isinstance(detDesc,str):
# #         # using same method for detector/descriptor
# #         featDet, descType = detectors(detDesc, numFeat)
# #         # featDesc, descType = descriptors(detDesc, numFeat)
# #
# #         kp0, des0 = featDet.detectAndCompute(im0, None)
# #         kp1, des1 = featDet.detectAndCompute(im1, None)
# #
# #     else:
# #         featDet, _ = detectors(detDesc[0], numFeat)
# #         featDesc, descType = descriptors(detDesc[1], numFeat)
# #
# #         kp0 = featDet.detect(im0, None)
# #         kp1 = featDet.detect(im1, None)
# #
# #         kp0, des0 = featDesc.compute(im0, kp0)
# #         kp1, des1 = featDesc.compute(im1, kp1)
# #
# #     if type=='cross':
# #         # BFMatcher with Cross Check
# #         if descType == 'binary':
# #             distType = 'Hamming'
# #             distMethod = cv.NORM_HAMMING
# #         else:
# #             distType = 'L2 Norm'
# #             distMethod = cv.NORM_L2
# #
# #         bfm = cv.BFMatcher(distMethod, crossCheck=True)
# #         matches = bfm.match(des0, des1)
# #         matches = sorted(matches, key=lambda x: x.distance)
# #
# #         title = '%s - matches: %s' % (distType, len(matches))
# #
# #         if output == 'figure' or output == 'figdata':
# #             resImg = cv.drawMatches(im0, kp0, im1, kp1, matches, np.array([]))
# #
# #         matches = [[m] for m in matches]
# #
# #     else:
# #         # BFMatcher with default params
# #         bfm = cv.BFMatcher()
# #         matches = bfm.knnMatch(des0, des1, k=2)
# #         title = 'Default - matches: %s' % len(matches)
# #
# #         if type=='ratio':
# #             # Apply ratio test
# #             good = []
# #             for m, n in matches:
# #                 if m.distance < ratio * n.distance:
# #                     good.append([m])
# #             matches = good
# #             title = 'Ratio - matches: %s, ratio: %s' % (len(matches), ratio)
# #
# #         if output == 'figure' or output == 'figdata':
# #             resImg = cv.drawMatchesKnn(im0, kp0, im1, kp1, matches, np.array([]), flags=2)
# #
# #     if output == 'figure' or output == 'figdata':
# #         plt.figure()
# #         plt.title(title)
# #         plt.imshow(resImg)
# #         plt.show()
# #
# #     if output == 'data' or output == 'figdata':
# #         img0_data = {'kp': kp0, 'des': des0}
# #         img1_data = {'kp': kp1, 'des': des1}
# #         return (matches, img0_data, img1_data)
# #
# #
# # def maxCompare(srcImg, dstImg, K, type='ratio', dd='sift', mthd='ransac', output='data', nf=0, rat=0.7):
# #     matches, srcData, dstData = bruteForce(srcImg, dstImg, type, dd, 'data', ratio=rat, numFeat=nf)
# #
# #     if isinstance(dd,str):
# #         det=dd
# #         desc=dd
# #     else:
# #         det=dd[0]
# #         desc=dd[1]
# #
# #     # Task C: Perform Robust Outlier Rejection
# #     src_pts = cv.KeyPoint_convert(srcData['kp'], [m[0].queryIdx for m in matches])
# #     dst_pts = cv.KeyPoint_convert(dstData['kp'], [m[0].trainIdx for m in matches])
# #
# #     if mthd == 'ransac':
# #         essentialMat, mask = cv.findEssentialMat(src_pts, dst_pts, K, cv.FM_RANSAC, 0.999, 1.0)
# #     elif mthd == 'lmeds':
# #         essentialMat, mask = cv.findEssentialMat(src_pts, dst_pts, K, cv.FM_LMEDS, 0.999)
# #     mask = mask.ravel().tolist()
# #     mask = [[msk] for msk in mask]
# #
# #     matchCmp = cv.drawMatchesKnn(srcImg, srcData['kp'], dstImg, dstData['kp'], matches, np.array([]), flags=2, matchesMask=mask)
# #
# #     finalMatches = sum([msk[0] for msk in mask])
# #
# #     print (finalMatches)
# #
# #     if output=='fig' or output=='figure' or output=='figdata':
# #         plt.figure()
# #         plt.imshow(matchCmp)
# #         if type=='':
# #             type = 'default'
# #         title = ('%s, %s, %s/%s - matches: %s' % (type, mthd, det, desc, finalMatches))
# #         if type=='ratio':
# #             title = ('%s, ratio: %s' % (title, str(rat)))
# #         plt.title(title)
# #
# #     if output=='data' or output=='figdata':
# #         return (essentialMat, src_pts, dst_pts, matches, mask)