from src.helpers.cfg import ThesisConfig
import numpy as np
import h5py
import os
import seaborn as sns
import matplotlib.pyplot as plt


# Determine model to use for evaluation
# configFiles = ['exp_configs/CNN_test_12.yaml', 'exp_configs/CNN_test_17.yaml']
# configFiles = ['exp_configs/trans_test_2.yaml', 'exp_configs/trans_test_4.yaml']
configFiles = ['exp_configs/scale_test_2.yaml', 'exp_configs/scale_test_4.yaml']
# evalType = 'val'
evalType = 'test'









# configFile = configFiles[0]
# config = ThesisConfig(configFile)
#
# # Import INS Data
# imuFilesDict = config.kittiPrepared['imu']
# truthFilesDict = config.kittiPrepared['truth']
#
# imuData = np.empty((0, 3))
# for seq in config.usedSeqs:
#     imuFile = config.getInputFiles(imuFilesDict, seq)
#     with h5py.File(imuFile, 'r') as f:
#         noisy_rot_xyz = np.array(f['noisy_rot_xyz'])
#     imuData = np.append(imuData, noisy_rot_xyz, axis=0)
#
# truthData = np.empty((0, 3))
# for seq in config.usedSeqs:
#     truthFile = config.getInputFiles(truthFilesDict, seq)
#     with h5py.File(truthFile, 'r') as f:
#         rot_xyz = np.array(f['rot_xyz'])
#     truthData = np.append(truthData, rot_xyz, axis=0)
#
# # get test idxs
# splitFilesDict = config.kittiPrepared['split']
# splitFile = config.getInputFiles(splitFilesDict)
# with h5py.File(splitFile, 'r') as f:
#     if evalType == 'test':
#         turnIdxs = np.array(f['testTurnIdxs'])
#         nonTurnIdxs = np.array(f['testNonTurnIdxs'])
#     elif evalType == 'val':
#         turnIdxs = np.array(f['valTurnIdxs'])
#         nonTurnIdxs = np.array(f['valNonTurnIdxs'])
# idxs = np.sort(np.concatenate((turnIdxs,nonTurnIdxs)))
#
#
# # get truth test rotations
# y_true_real = truthData[idxs,:]
#
# # get imu test rotations
# y_pred_real = imuData[idxs,:]
#
#
# # calculate errors
# errors = y_true_real - y_pred_real
# rmseErrors = np.sqrt(np.mean(errors**2,axis=0)) # dx,dy,dz,X,Y,Z
# meanErrors = np.mean(errors,axis=0)
# stdErrors = np.std(errors,axis=0)
# rot_format = 'dx: %8f, dy: %8f, dz: %8f'
# rotErrRMSE_deg, transErrRMSE_meters = rmseErrors[:3] * 180 / np.pi, rmseErrors[3:]
# rotErrMean_deg, transErrMean_meters = meanErrors[:3] * 180 / np.pi, meanErrors[3:]
# rotErrStd_deg, transErrStd_meters = stdErrors[:3] * 180 / np.pi, stdErrors[3:]
# print('Rotation Error RMSE (deg):        %s' % (rot_format % tuple(rotErrRMSE_deg)))
# print('Rotation Error Mean (deg):        %s' % (rot_format % tuple(rotErrMean_deg)))
# print('Rotation Error Std  (deg):        %s' % (rot_format % tuple(rotErrStd_deg)))


for configFile in configFiles:
    config = ThesisConfig(configFile)



    # Parameters
    name = config.expName
    numOutputs      = config.modelParms['numOutputs']

    # Get Files
    evalFilesDict = config.resultPaths['evaluations']
    figFilesDict = config.resultPaths['figures']

    figFolder = config.getFolderRef(config.getOutputFiles(figFilesDict, True))




    # Get Predictions Save File
    evalFolder = ''
    for pathSection in evalFilesDict['dir']:
        evalFolder = os.path.join(evalFolder, pathSection)
    predictionsFile = os.path.join(evalFolder, evalType+'_predictions.hdf5')


    if os.path.exists(predictionsFile):

        # Load Predictions
        with h5py.File(predictionsFile, 'r') as f:
            y_pred_real = np.array(f['predictions'])
            y_true_real = np.array(f['truth'])
            evalType = str(np.array(f['evalType']))
            min_val_epoch = np.array(f['epoch'])
            min_val_loss = np.array(f['valLossAtEpoch'])



        print('Min Validation Loss: %s, Epoch %s' % (min_val_loss, min_val_epoch))


        # Plot True vs. Predicted

        title = 'True vs. Prediction'
        rot_types = ['Rotation delta_X',
                     'Rotation delta_Y',
                     'Rotation delta_Z']
        trans_types = ['Translation X',
                       'Translation Y',
                       'Translation Z']
        scale_types = ['Translation Scale']
        rot_units = 'rad'
        trans_units = 'meters'

        title_types = np.empty(0)
        units = np.empty(0)
        if numOutputs==1:
            title_types = scale_types
            units = [trans_units]
        elif numOutputs==3:
            title_types = trans_types
            units = np.append(units, [trans_units]*3)
        else:
            title_types = np.append(title_types, rot_types)
            title_types = np.append(title_types, trans_types)
            units = np.append(units, [rot_units]*3)
            units = np.append(units, [trans_units]*3)







        # Calculate average loss in each direction
        errors = y_true_real - y_pred_real
        rmseErrors = np.sqrt(np.mean(errors**2,axis=0)) # dx,dy,dz,X,Y,Z
        meanErrors = np.mean(errors,axis=0)
        stdErrors = np.std(errors,axis=0)

        rot_format = 'dx: %8f, dy: %8f, dz: %8f'
        trans_format = ' x: %8f,  y: %8f,  z: %8f'
        scale_format = ' s: %8f'
        if numOutputs==1:
            scaleErrRMSE_meters = rmseErrors
            scaleErrMean_meters = meanErrors
            scaleErrStd_meters = stdErrors
            print('Translation Error RMSE (meters):  %s' % (scale_format % tuple(scaleErrRMSE_meters)))
            print('Translation Error Mean (meters):  %s' % (scale_format % tuple(scaleErrMean_meters)))
            print('Translation Error Std (meters):   %s' % (scale_format % tuple(scaleErrStd_meters)))
        elif numOutputs==3:
            transErrRMSE_meters = rmseErrors
            transErrMean_meters = meanErrors
            transErrStd_meters = stdErrors
            print('Translation Error RMSE (meters):  %s' % (trans_format % tuple(transErrRMSE_meters)))
            print('Translation Error Mean (meters):  %s' % (trans_format % tuple(transErrMean_meters)))
            print('Translation Error Std (meters):   %s' % (trans_format % tuple(transErrStd_meters)))
        else:
            rotErrRMSE_deg, transErrRMSE_meters = rmseErrors[:3] * 180 / np.pi, rmseErrors[3:]
            rotErrMean_deg, transErrMean_meters = meanErrors[:3] * 180 / np.pi, meanErrors[3:]
            rotErrStd_deg, transErrStd_meters = stdErrors[:3] * 180 / np.pi, stdErrors[3:]
            print('Rotation Error RMSE (deg):        %s' % (rot_format % tuple(rotErrRMSE_deg)))
            print('Rotation Error Mean (deg):        %s' % (rot_format % tuple(rotErrMean_deg)))
            print('Rotation Error Std  (deg):        %s' % (rot_format % tuple(rotErrStd_deg)))
            print('Translation Error RMSE (meters):  %s' % (trans_format % tuple(transErrRMSE_meters)))
            print('Translation Error Mean (meters):  %s' % (trans_format % tuple(transErrMean_meters)))
            print('Translation Error Std (meters):   %s' % (trans_format % tuple(transErrStd_meters)))





















        lines = []
        for i in range(y_true_real.shape[1]):
            truth = y_true_real[:,i]
            preds = y_pred_real[:,i]
            min_val = np.max((np.min(truth),np.min(preds)))
            max_val = np.min((np.max(truth),np.max(preds)))
            # min_val = np.min((truth,preds))
            # max_val = np.max((truth,preds))
            min_dec_places = -int(np.round(np.log10(np.abs(min_val))))+1
            max_dec_places = -int(np.round(np.log10(np.abs(max_val))))+1
            if min_dec_places<0:
                min_dec_places = 0
            if max_dec_places<0:
                max_dec_places = 0
            line_min = np.round(min_val-0.5*10**-min_dec_places,decimals=min_dec_places)
            line_max = np.round(max_val+0.5*10**-max_dec_places,decimals=max_dec_places)
            line = np.array([line_min, line_max])
            lines.append(line)



        # for i in range(y_true_real.shape[1]):
        #     plt.figure()
        #     plt.scatter(y_true_real[:, i], y_pred_real[:, i])
        #     plt.plot(lines[i], lines[i], c='r')
        #     plt.title('%s -- %s %s' % (name, title, title_types[i]))
        #     plt.xlabel('True (%s)' % units[i])
        #     plt.ylabel('Predicted (%s)' % units[i])
        #     plt.axis('equal')
        #     saveFile = '%s.png' % os.path.join(figFolder,evalType.upper() + ' ' + title_types[i])
        #     plt.savefig(saveFile)


        for i in range(y_true_real.shape[1]):
            plt.figure()
            # sns.jointplot(x=y_true_real[:,i], y=y_pred_real[:,i], s=200, kind='kde')
            sns.kdeplot(y_true_real[:, i], y_pred_real[:, i], gridsize=50, shade_lowest=False)#, cmap='Blues',gridsize=1000)#, shade=True)#, shade_lowest=True)#, cmap='Reds', shade=True)
            plt.plot(y_true_real[:, i], y_pred_real[:, i], linestyle='', marker='.', markersize=0.5)
            plt.plot(lines[i], lines[i], c='r')
            plt.title('%s -- %s %s' % (name, title, title_types[i]))
            plt.xlabel('True (%s)' % units[i])
            plt.ylabel('Predicted (%s)' % units[i])
            plt.axis('equal')
            saveFile = '%s.png' % os.path.join(figFolder,evalType.upper() + ' ' + title_types[i])
            plt.savefig(saveFile)
plt.show(block=True)