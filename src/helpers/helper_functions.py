# import matplotlib
# if matplotlib.get_backend() is not 'TkAgg':
#     matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import os
import keras
import h5py
import numpy as np



from src.helpers.cfg import ThesisConfig



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

# def plotLoss(hist, name, noVal=False, start=0, end=None):
#     loss = hist['loss']
#     lr_changes = hist['lr_changes']
#     lr_changes = lr_changes[np.where(lr_changes>start)[0]]
#     if end is None:
#         end = len(loss)
#     if len(loss)>start:
#         plt.plot(range(start, end), loss[start:end], 'bo', label='Training loss')
#         if not noVal:
#             val_loss = hist['val_loss']
#             plt.plot(range(start, end), val_loss[start:end], 'r', label='Validation loss')
#         # TODO: Plot changes on monitor loss
#         plt.plot(lr_changes-1, loss[lr_changes-1], 'ko', label='LR Adjustments')
#         plt.title('Training and validation loss, %s' % name)
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss (MSE)')

def round(values, decimals=2, dir='up'):
    add_amt = 1.5*(10**(-decimals))
    if dir == 'down':
        add_amt = -add_amt
    return np.round(values + add_amt, decimals)


def getOptimizer(type, **args):
    if (type=='RMSprop'):
        modelOptimizer = keras.optimizers.RMSprop(lr=args['lr'])
    return modelOptimizer


def getNormParms(data, trainIdxs):
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

def getH5pyData(configClass, filesDict, sequences, key, numOutputs):
    data = np.empty((0, numOutputs))
    for seq in sequences:
        h5pyFile = configClass.getInputFiles(filesDict, seq)
        if h5pyFile is None:
            sys.exit()
        with h5py.File(h5pyFile, 'r') as f:
            seqData = np.array(f[key])
        data = np.append(data, seqData, axis=0)
    return data