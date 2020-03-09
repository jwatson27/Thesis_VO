# import matplotlib
# if matplotlib.get_backend() is not 'TkAgg':
#     matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import os
import keras
import h5py
import numpy as np
import cv2 as cv


from src.helpers.cfg import ThesisConfig
from src.helpers.coframes import cvtToDcm_sd, cvtToRpy_sd


def save_model(model, save_dir=os.path.join(os.getcwd(), 'saved_models'),
               model_file_name='keras_cifar10_trained_model.h5'):
    """
    Save model and current weights
    :param model: Keras model
    :param save_dir: path name to save directory
    :param model_file_name: filename for saved model
    :return: nothing
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_file_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def load_model(save_dir, model_file_name):
    # Load model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_file_name)
    model = keras.models.load_model(model_path)
    print('Loaded trained model from %s ' % model_path)
    return model


def save_history(history, history_file_name, save_dir=os.path.join(os.getcwd(), 'saved_models'), noVal=False):
    """
    Save model and current weights
    :param model: Keras model
    :param save_dir: path name to save directory
    :param model_file_name: filename for saved model
    :return: nothing
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    history_path = os.path.join(save_dir, history_file_name)
    with h5py.File(history_path, 'w') as f:
        os.chmod(history_path, 0o666)
        f.create_dataset('loss', data=history['loss'])
        # f.create_dataset('acc', data=history['acc'])
        if not noVal:
            f.create_dataset('val_loss', data=history['val_loss'])
            # f.create_dataset('val_acc', data=history['val_acc'])
        f.create_dataset('lr_changes', data=history['lr_changes'])
    print('Saved model history at %s ' % history_path)

def load_history(history_file_name, save_dir, noVal=False):
    # Load model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    history_path = os.path.join(save_dir, history_file_name)
    history = {}
    with h5py.File(history_path, 'r') as f:
        history['loss'] = np.array(f['loss'])
        # history['acc'] = np.array(f['acc'])
        if not noVal:
            history['val_loss'] = np.array(f['val_loss'])
            # history['val_acc'] = np.array(f['val_acc'])
        history['lr_changes'] = np.array(f['lr_changes'])
    print('Loaded trained model from %s ' % history_path)
    return history

def hms_time(time_delta):
    seconds = time_delta.seconds
    hours = seconds // 3600
    seconds -= hours * 3600
    minutes = seconds // 60
    seconds -= minutes * 60
    return(hours, minutes, seconds)

def round(values, decimals=2, dir='up'):
    add_amt = 1.5*(10**(-decimals))
    if dir == 'down':
        add_amt = -add_amt
    return np.round(values + add_amt, decimals)


def getOptimizer(type, **args):
    if (type=='RMSprop'):
        modelOptimizer = keras.optimizers.RMSprop(lr=args['lr'])
    if (type=='Adam'):
        modelOptimizer = keras.optimizers.Adam(learning_rate=args['lr'])
    return modelOptimizer


def calcNormParms(data, trainIdxs):
    trainData = data[trainIdxs]
    mean = np.mean(trainData, axis=0)
    std = np.std(trainData, axis=0)
    return (mean, std)

def applyNorm(data, normParms):
    mean, std = normParms
    normalized = (data - mean) / std
    return normalized

def undoNorm(normalized, normParms):
    mean, std = normParms
    data = (normalized * std) + mean
    return data

def loadNormParms(normParmsFile, name):
    with h5py.File(normParmsFile, 'r') as f:
        return(np.array(f[name]))

def getH5pyData(configClass, filesDict, sequences, key, numOutputs, camera=None):
    data = np.empty((0, numOutputs))
    for seq in sequences:
        h5pyFile = configClass.getInputFiles(filesDict, seq, camera)
        if h5pyFile is None:
            sys.exit()
        with h5py.File(h5pyFile, 'r') as f:
            seqData = np.array(f[key])
        data = np.append(data, seqData, axis=0)
    return data



def getKpsAndDescs(detector, imageName):
    image = cv.cvtColor(cv.imread(imageName), cv.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return (keypoints, descriptors)

def matchAndRatioTest(sourceDesc, destinationDesc, ratio=0.7):
    # brute force matcher with default params
    bfm = cv.BFMatcher()
    matches = bfm.knnMatch(sourceDesc, destinationDesc, k=2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    matches = good

    return matches

def outlierRejectionRANSAC(srcKps_matched, dstKps_matched, camCalMat):
    essentialMat, mask = cv.findEssentialMat(srcKps_matched, dstKps_matched, camCalMat, cv.FM_RANSAC, 0.999, 1.0)
    mask = mask.ravel().tolist()
    mask = np.array([[msk] for msk in mask])
    return (essentialMat, mask)




def compareTruthToEpipolar(configFile, seq, cam, idxList=None, type='both'):

    config = ThesisConfig(configFile)

    # get epipolar data
    epiFilesDict = config.kittiPrepared['epipolar']
    epiFile = config.getInputFiles(epiFilesDict, seq, cam)
    with h5py.File(epiFile, 'r') as f:
        epi_rot_xyz = np.array(f['epi_rot_xyz'])
        epi_trans_xyz = np.array(f['epi_trans_xyz'])

    # get truth data
    truthFilesDict = config.kittiPrepared['truth']
    truthFile = config.getInputFiles(truthFilesDict, seq)
    with h5py.File(truthFile, 'r') as f:
        true_rot_xyz = np.array(f['rot_xyz'])
        true_trans_xyz = np.array(f['trans_xyz'])

    if idxList is None:
        idxList = range(len(true_rot_xyz))

    for img_idx in idxList:
        if type=='rot' or type=='both':
            epi_rot = epi_rot_xyz[img_idx] * 180 / np.pi
            true_rot = true_rot_xyz[img_idx] * 180 / np.pi
            print('Epi Rot  [%d]: [%12f, %12f, %12f]' % (img_idx, epi_rot[0], epi_rot[1], epi_rot[2]))
            print('True Rot [%d]: [%12f, %12f, %12f]' % (img_idx, true_rot[0], true_rot[1], true_rot[2]))
        if type=='trans' or type=='both':
            epi_trans = epi_trans_xyz[img_idx]
            true_trans = true_trans_xyz[img_idx]
            true_trans = true_trans/np.sqrt(np.sum(true_trans**2))  # Convert to unit vector
            print('Epi Trans  [%d]: [%12f, %12f, %12f]' % (img_idx, epi_trans[0], epi_trans[1], epi_trans[2]))
            print('True Trans [%d]: [%12f, %12f, %12f]' % (img_idx, true_trans[0], true_trans[1], true_trans[2]))


def getBestValLoss(config):
    history_filename = config.trainingParms['histFilename']
    historyFilesDict = config.trainPaths['history']
    historyFiles = config.getInputFiles(historyFilesDict)
    historyFolder = config.getFolderRef(historyFiles)
    history_filepath = os.path.join(historyFolder, history_filename)

    with h5py.File(history_filepath, 'r') as f:
        epochs = np.array(f['epochs'], dtype=np.int)
        numEpochs = len(epochs)
        if 'val_loss' in f:
            val_loss = np.array(f['val_loss'])
            min_val_loss = np.nanmin(val_loss)
            min_val_loss_epoch = (epochs[np.nanargmin(val_loss)])

    return (min_val_loss, min_val_loss_epoch)


def getValLoss(config, epochNum=None):
    history_filename = config.trainingParms['histFilename']
    historyFilesDict = config.trainPaths['history']
    historyFiles = config.getInputFiles(historyFilesDict)
    historyFolder = config.getFolderRef(historyFiles)
    history_filepath = os.path.join(historyFolder, history_filename)

    with h5py.File(history_filepath, 'r') as f:
        epochs = np.array(f['epochs'], dtype=np.int)
        if 'val_loss' in f:
            val_loss = np.array(f['val_loss'])

            if epochNum is None:
                val_loss_at_epoch = np.nanmin(val_loss)
                epoch = (epochs[np.nanargmin(val_loss)])
            else:
                val_loss_at_epoch = val_loss[epochNum-1]
                epoch = epochNum

    return (val_loss_at_epoch, epoch)




def getFolder(dirList):
    folder = ''
    for pathSection in dirList:
        folder = os.path.join(folder, pathSection)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder



