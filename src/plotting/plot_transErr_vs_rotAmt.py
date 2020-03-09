from src.helpers.cfg import ThesisConfig
import numpy as np
import h5py
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.helpers.eval_helpers import loadPredictions

# Determine model to use for evaluation
# configFiles = ['exp_configs/CNN_test_12.yaml', 'exp_configs/CNN_test_17.yaml']
# configFiles = ['exp_configs/trans_test_2.yaml', 'exp_configs/trans_test_4.yaml']
# configFiles = ['exp_configs/scale_test_2.yaml', 'exp_configs/scale_test_4.yaml']
# evalType = 'val'
evalType = 'test'

tests = ['CNN_test_17', 'CNN_test_12']
# tests = ['trans_test_4', 'trans_test_2']
# tests = ['scale_test_4', 'scale_test_2']


model_type = ['base', 'aided']


# test = tests[1]
colors = ['r', 'g', 'b']
markers = ['p', 'o']
marker_sizes = [0.8, 1]

errors_list, preds_list, truth_list, filetruthrots_list = loadPredictions(tests, evalType)


file_truth_rots = filetruthrots_list[0]
truth = truth_list[0]
out_types = ['rotX', 'rotY', 'rotZ', 'traX', 'traY', 'traZ']








for i in range(preds_list[0].shape[1]):
    plt.figure()
    plt.plot(truth[:, i], label='truth_' + model_type[0])
    for j in range(len(preds_list)):
        plt.plot(preds_list[j][:, i], label='pred_' + model_type[j])
    plt.title(out_types[i])
    plt.legend()



filetruth_sort_idxs = []
for i in range(file_truth_rots.shape[1]):
    filetruth_sort_idxs.append(np.argsort(file_truth_rots[:,i]))

truth_sort_idxs = []
for i in range(truth.shape[1]):
    truth_sort_idxs.append(np.argsort(truth[:,i]))



# SORTED BY RESPECTIVE OUTPUT
for i in range(preds_list[0].shape[1]):
    sort_idxs = truth_sort_idxs[i]
    plt.figure()
    for j in range(len(preds_list)):
        plt.plot(truth[sort_idxs,i], preds_list[j][sort_idxs, i], label='pred_' + model_type[j])
    plt.title(out_types[i] + '_sorted')
    plt.plot(truth[sort_idxs, i], truth[sort_idxs, i], c='k', label='truth')
    plt.legend()





# SORTED BY YAW
for i in range(preds_list[0].shape[1]):
    sort_idxs = filetruth_sort_idxs[1]
    plt.figure()
    plt.plot(file_truth_rots[sort_idxs,1], truth[sort_idxs, i], label='truth_' + model_type[0])
    for j in range(len(preds_list)):
        plt.plot(file_truth_rots[sort_idxs,1], preds_list[j][sort_idxs, i], label='pred_' + model_type[j])
    plt.title(out_types[i] + '_yrotSorted')
    plt.legend()



for i in range(preds_list[0].shape[1]):
    plt.figure()
    plt.plot(truth[:, i], label='truth_' + model_type[0])
    for j in range(len(preds_list)):
        plt.plot(preds_list[j][:, i], label='pred_' + model_type[j])
    plt.title(out_types[i])
    plt.legend()






plt.show(block=True)

# # Plot True vs. Predicted
#
# title = 'True vs. Prediction'
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
# if numOutputs == 1:
#     title_types = scale_types
#     units = [trans_units]
# elif numOutputs == 3:
#     title_types = trans_types
#     units = np.append(units, [trans_units] * 3)
# else:
#     title_types = np.append(title_types, rot_types)
#     title_types = np.append(title_types, trans_types)
#     units = np.append(units, [rot_units] * 3)
#     units = np.append(units, [trans_units] * 3)


# plt.figure()
# rotAmt = truthRots * 180 / np.pi
# if numOutputs==6:
#     # Get translation error
#     rotErrors = errors[:,:3]
#     transErrors = errors[:,3:]
# elif numOutputs==3:
#     transErrors = errors
# else:
#     transErrors = errors


# # compare = rotErrors
# # compare = transErrors[:,2]
# for j in range(compare.shape[1]):
#     plt.figure(figs[j].number)
#     plt.plot(compare[:,j])
#     plt.plot(y_true_real[:, j], label='truth')
#     # plt.plot(rotAmt[:,1],compare[:,j], linestyle='', c=colors[i], marker=markers[i], markersize=marker_sizes[i])