import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from src.helpers.helper_functions import load_history, plotLoss, round


print('Save Loss Figure')
# modelName = 'WatsonCNN_2_model_1'         # best_loss: 0.0014    best_val_loss: 0.1599        (0.001, 281, drop=(None, None))
# modelName = 'WatsonCNN_2_model_2'         # best_loss: 0.1900    best_val_loss: 0.1985        (0.001, 107, drop=(0.5,0.5))
# modelName = 'WatsonCNN_2_model_3'         # best_loss: 0.0479    best_val_loss: 0.1546        (0.001, 201, drop=(0.2,0.2))
# modelName = 'WatsonCNN_2_model_withIMU_0' # best_loss: 0.0848    best_val_loss: 0.1729        (0.001, 200, drop=(0.2,0.2))
# modelName = 'WatsonCNN_2_model_4'         # best_loss: 0.0270    best_val_loss: 0.1284        (10^-4, 43, drop=(None, None))
# modelName = 'WatsonCNN_2_model_5'         # best_loss: 0.1390    best_val_loss: 0.1511        (10^-4, 49, drop=(0.5,0.5))
# modelName = 'WatsonCNN_2_model_6'         # best_loss: 0.0352    best_val_loss: 0.1256        (10^-4, 90, drop=(0.2,0.2))
# modelName = 'WatsonCNN_2_model_7'         # best_loss: 0.0233    best_val_loss: 0.1252        (10^-4, 52, drop=(None,0.4))
# modelName = 'WatsonCNN_2_model_8'         # best_loss: 0.0840    best_val_loss: 0.1296        (10^-4, 70, drop=(0.4,None))
# modelName = 'WatsonCNN_2_model_10'        # best_loss: 0.0290    best_val_loss: 0.1258        (10^-4, 70, drop=(0.1,0.3))
# modelName = 'WatsonCNN_2_model_11'        # best_loss: 0.0300    best_val_loss: 0.1220        (10^-4, 72, drop=(0.1,0.4))
# modelName = 'WatsonCNN_2_model_13'        # best_loss: 0.0592    best_val_loss: 0.1239        (10^-4, 55, drop=(0.1,0.8))
# modelName = 'WatsonCNN_2_model_14'        # best_loss: 0.1098    best_val_loss: 0.1496        (10^-4, 86, drop=(0.1,0.95))
# modelName = 'WatsonCNN_2_model_withIMU_2'        # best_loss: 0.0013    best_val_loss: 0.1328        (10^-4, 100, drop=(None,None))
# modelName = 'WatsonCNN_2_model_12'        # best_loss: 0.0082    best_val_loss: 0.1118,   168     (10^-4, 190, drop=(0.1,0.5))
modelName = 'WatsonCNN_2_model_withIMU_1'        # best_loss: 0.0103    best_val_loss: 0.1118,    158     (10^-4, 190, drop=(0.1,0.5))
save_dir = os.path.join(os.getcwd(), 'saved_models', modelName)
metaSaveFile = '/'.join((save_dir, 'metadata.npy'))
metadata = np.load(metaSaveFile, allow_pickle=True).item()
history_file_name = 'training_history.hdf5'
history = load_history(history_file_name, save_dir)
loss = history['loss']
numEpochs = len(loss)
best_train_epoch = np.argmin(loss)
val_loss = history['val_loss']
best_val_epoch = np.argmin(val_loss)
figPath = os.path.join(save_dir, 'loss_results.png')
print('\n')
print('num_epochs:    %6d' % numEpochs)
print('best_loss:     %0.4f' % loss[best_train_epoch])
print('best_val_loss: %0.4f' % val_loss[best_val_epoch])
print('best_val_epoch: %s' % best_val_epoch)
print()

plt.figure()
lossLessThanThresh = np.where(loss < 1)[0]
startEpoch = lossLessThanThresh[0]
ymax = round(np.max(loss[lossLessThanThresh]), dir='up')
ymin = round(np.min(loss[lossLessThanThresh]), dir='down')
valmax = round(np.max(val_loss[lossLessThanThresh]), dir='up')
valmin = round(np.min(val_loss[lossLessThanThresh]), dir='down')
if valmax > ymax:
    ymax = valmax
if valmin < ymin:
    ymin = valmin
endEpoch = None
plotLoss(history, name=modelName, start=startEpoch, end=endEpoch)
plt.plot(best_val_epoch, val_loss[best_val_epoch], 'go', label='Best Val Loss')
plt.legend()
plt.ylim(bottom=ymin, top=ymax)
plt.savefig(figPath)
plt.show()

# for modelNum in range(6):
#     modelName = 'CNN_model_new_%s' % modelNum
#     save_dir = os.path.join(os.getcwd(), 'saved_models', modelName)
#     history = load_history(history_file_name, save_dir)
#     val_loss = history['val_loss']
#     best_epoch = np.argmin(val_loss)
#     best_val_loss = val_loss[best_epoch]
#     print('Model %s, Best Loss (Epoch %s): %0.6f' % (modelName, best_epoch, best_val_loss))
#
# plt.show()