import matplotlib
if matplotlib.get_backend() is not 'TkAgg':
    matplotlib.use('TkAgg')

import os
import numpy as np
import h5py
import keras
import matplotlib.pyplot as plt
from src.helpers.helper_functions import round



# TODO: Add Custom Callback for Timing Display
# Initialization
    # tz = 'US/Eastern'
    # start_time = datetime.now(timezone(tz))
# In Loop
    # curr_time = datetime.now(timezone(tz))
    # elapsed = '%02d:%02d:%02d' % hms_time(curr_time - start_time)
    # curr_time = curr_time.strftime('%H:%M:%S')


# TODO: Add Learning Rate Changes to Plot
# print('Current LR: %s' % K.eval(model.optimizer.lr))
# monitor = reduceLRdict['monitor']
# patience = reduceLRdict['patience']
# hist = history[monitor]
# last_lr_change = lr_changes[-1]
# epochs_since_last_change = training_iteration - last_lr_change
# print('epochs_since_last_change: %s' % epochs_since_last_change)
# if epochs_since_last_change >= patience:
#     # print('Previous %s: %s, Most Recent %s: %s' % (monitor, hist[-(patience+1)], monitor, hist[-1]))
#     if abs(hist[-(patience+1)] - hist[-1]) < reduceLRdict['min_delta']:
#         old_LR = K.eval(model.optimizer.lr)
#         new_LR = old_LR*reduceLRdict['factor']
#         lossFunc = model.loss_functions[0]
#         model.compile(optimizer=optimizers.RMSprop(lr=new_LR), loss=lossFunc)
#         lr_changes.append(training_iteration)
#         print('Decreased LR from %s to %s' % (old_LR, new_LR))


class SaveHistoryToFile(keras.callbacks.Callback):
    def __init__(self, filepath, save_accuracy=False, **kargs):
        super().__init__(**kargs)
        self.filepath      = filepath
        self.saveFolder    = os.path.dirname(filepath)
        self.save_accuracy = save_accuracy

        self.epochs     = np.array([])
        self.train_loss = np.array([])
        self.val_loss   = np.array([])
        self.train_acc  = np.array([])
        self.val_acc    = np.array([])

    # things done on beginning of epoch
    def on_epoch_begin(self, epoch, logs={}):
        if os.path.exists(self.filepath):
            with h5py.File(self.filepath, 'r') as f:
                epochs          = np.array(f['epochs'])
                curr_epoch_list = np.where(epochs == epoch)[0]
                if curr_epoch_list:
                    curr_epoch_idx = curr_epoch_list[0]
                    self.epochs = epochs[:curr_epoch_idx+1]
                    self.train_loss = np.array(f['loss'])[:curr_epoch_idx+1]
                    self.val_loss   = np.array(f['val_loss'])[:curr_epoch_idx+1]
                    if self.save_accuracy:
                        self.train_acc = np.array(f['acc'])[:curr_epoch_idx+1]
                        self.val_acc   = np.array(f['val_acc'])[:curr_epoch_idx+1]

        else:
            if not os.path.exists(self.saveFolder):
                os.makedirs(self.saveFolder)
                print()
                print('Created Folder: %s' % self.saveFolder)
                print()

    # things done on end of the epoch
    def on_epoch_end(self, epoch, logs={}):
        self.epochs     = np.append(self.epochs,     epoch+1)
        self.train_loss = np.append(self.train_loss, logs.get('loss'))
        self.val_loss   = np.append(self.val_loss,   logs.get('val_loss'))
        self.train_acc  = np.append(self.train_acc,  logs.get('acc'))
        self.val_acc    = np.append(self.val_acc,    logs.get('val_acc'))

        with h5py.File(self.filepath, 'w') as f:
            f.create_dataset('epochs',   data=self.epochs)
            f.create_dataset('loss',     data=self.train_loss)
            f.create_dataset('val_loss', data=self.val_loss)
            if self.save_accuracy:
                f.create_dataset('acc',     data=self.train_acc)
                f.create_dataset('val_acc', data=self.val_acc)





class PlotHistory(keras.callbacks.Callback):
    def __init__(self, filepath, plotLossFigure=True,
                 monitorList=None,
                 windowSize=None,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 borderMult=0.2,
                 blockProcessing=False,
                 **kargs):
        super().__init__(**kargs)

        self.filepath = filepath
        self.plotLossFigure = plotLossFigure

        if monitorList is None:
            if plotLossFigure:
                monitorList = ['loss', 'val_loss']
            else:
                monitorList = ['acc', 'val_acc']
        self.monitor_list = monitorList

        if title is None:
            title = 'Training and Validation'
            if plotLossFigure:
                title = '%s Loss' % title
            else:
                title = '%s Accuracy' % title
        self.title = title

        if xlabel is None:
            xlabel = 'Epochs'
        self.xlabel = xlabel

        if ylabel is None:
            if plotLossFigure:
                ylabel = 'Loss'
            else:
                ylabel = 'Accuracy'
        self.ylabel = ylabel

        self.windowSize = windowSize
        self.borderMult = borderMult

        fig = plt.figure()
        plt.ion()
        plt.show()
        self.figureNum = fig.number
        self.legendPlotted = False
        self.blockProcessing = blockProcessing


    # things done on beginning of epoch
    def on_epoch_begin(self, epoch, logs={}):
        return

    # things done on end of the epoch
    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists(self.filepath):
            with h5py.File(self.filepath, 'r') as f:
                history = {}
                for key in f:
                    history[key] = np.array(f[key])
                self.history = history

            if self.figExists():
                self.plotHistory(epoch)
                self.updateWindowSize(epoch)
                if self.blockProcessing:
                    plt.show(block=True)
                else:
                    plt.draw()
                plt.pause(0.001)

    def figExists(self):
        return plt.fignum_exists(self.figureNum)

    def plotHistory(self, currEpoch):
        if self.plotLossFigure:
            self.plotLoss(currEpoch)
        else:
            self.plotAccuracy(currEpoch)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if not self.legendPlotted:
            plt.legend()
            self.legendPlotted = True

    def updateWindowSize(self, currEpoch):
        windowSize = self.windowSize
        epochs = self.history['epochs'][:currEpoch]
        numEpochs = len(epochs)
        borderMult = self.borderMult

        if self.plotLossFigure:
            values = self.lossValues
        else:
            values = self.accValues

        xmax = numEpochs
        xmin = epochs[0]
        if (not windowSize is None) and (numEpochs > windowSize):
            values = values[:,-windowSize:]
            xmin = numEpochs-windowSize

        ymax = round(np.max(values), dir='up')
        ymin = round(np.min(values), dir='down')

        xborder = borderMult*(xmax-xmin)
        yborder = borderMult*(ymax-ymin)

        plt.xlim(left=xmin-xborder, right=xmax+xborder)
        plt.ylim(bottom=ymin-yborder, top=ymax+yborder)

    def plotLoss(self, currEpoch):
        epochs = self.history['epochs'][:currEpoch]
        lossTypes = ['loss', 'val_loss']
        lossValues = np.empty(shape=(0,len(epochs)))
        for monitor in self.monitor_list:
            if (monitor in lossTypes) and (monitor in self.history):
                colorFmt = self.colorFormat(monitor)
                dataLabel = self.dataLabel(monitor)
                plt.plot(epochs, self.history[monitor][:currEpoch], colorFmt, label=dataLabel)
                lossValues = np.append(lossValues, [self.history[monitor][:currEpoch]], axis=0)
        self.lossValues = lossValues

    def plotAccuracy(self, currEpoch):
        epochs = self.history['epochs'][:currEpoch]
        accTypes = ['acc', 'val_acc']
        accValues = np.empty(shape=(0,len(epochs)))
        for monitor in self.monitor_list:
            if (monitor in accTypes) and (monitor in self.history):
                colorFmt = self.colorFormat(monitor)
                dataLabel = self.dataLabel(monitor)
                plt.plot(epochs, self.history[monitor][:currEpoch], colorFmt, label=dataLabel)
                accValues = np.append(accValues, [self.history[monitor][:currEpoch]], axis=0)
        self.accValues = accValues

    def dataLabel(self, monitor):
        if monitor=='loss':
            return ('Training Loss')
        elif monitor=='val_loss':
            return ('Validation Loss')
        elif monitor=='acc':
            return ('Training Accuracy')
        elif monitor=='val_acc':
            return ('Validation Accuracy')

    def colorFormat(self, monitor):
        if monitor=='loss' or monitor=='acc':
            return ('bo')
        elif monitor=='val_loss' or monitor=='val_acc':
            return ('r')



