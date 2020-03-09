import matplotlib
if matplotlib.get_backend() is not 'TkAgg':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import norm, binned_statistic




def getTitleTypesAndUnits(numOutputs):
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
    if numOutputs == 1:
        title_types = scale_types
        units = [trans_units]
    elif numOutputs == 3:
        title_types = trans_types
        units = np.append(units, [trans_units] * 3)
    else:
        title_types = np.append(title_types, rot_types)
        title_types = np.append(title_types, trans_types)
        units = np.append(units, [rot_units] * 3)
        units = np.append(units, [trans_units] * 3)

    return (title_types, units)




def plotTrueVsPredAndCounts(true, pred, numBins=500, minNumPerBin=10, name=None, title=None, title_type=None, units=None, saveFile=None):
    counts, edges, _ = binned_statistic(true, pred, statistic='count', bins=numBins)
    mean, _, _ = binned_statistic(true, pred, statistic='mean', bins=numBins)
    std, _, _ = binned_statistic(true, pred, statistic='std', bins=numBins)

    midpoints = (edges[1:] + edges[:-1]) / 2
    valid = np.array(counts >= minNumPerBin, np.float)
    for j, val in enumerate(valid):
        if not val:
            valid[j] = np.nan
    mean = mean * valid
    std = std * valid

    fig = plt.figure()
    gridspec.GridSpec(4, 1)

    topTitle = ''
    if name is not None:
        topTitle += name
    if title is not None or title_type is not None:
        if topTitle:
            topTitle += ' --'
    if title is not None:
        topTitle += ' ' + title
    if title_type is not None:
        topTitle += ' ' + title_type
    fig.suptitle(topTitle)

    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax1.scatter(true, pred, label='data')
    ax1.plot(midpoints, mean, 'r', label='mean')
    ax1.plot(midpoints, mean + std, 'k', label='+/- stdev')
    ax1.plot(midpoints, mean - std, 'k')
    # ax1.set_xlabel('Bin Midpoints (%s)' % units)
    lab = 'Predicted'
    if units is not None:
        lab += ' (%s)' % units
    ax1.set_ylabel(lab)
    ax1.axis('equal')
    plt.legend()

    fig.canvas.draw()
    common_xlim = plt.xlim()

    ax2 = plt.subplot2grid((4, 1), (3, 0))
    plt.setp(ax2, xlim=common_xlim)
    ax2.plot(midpoints, counts, label='num points')
    ax2.plot(midpoints, np.ones(midpoints.shape) * minNumPerBin, 'k', label='min per bin')
    lab = 'Bin Midpoints'
    if units is not None:
        lab += ' (%s)' % units
    ax2.set_xlabel(lab)
    ax2.set_ylabel('Bin Counts')
    plt.legend()

    if saveFile is not None:
        plt.savefig(saveFile)







def plotHistAndGuassian(data, numBins=500, name=None, title=None, title_type=None, units=None, saveFile=None):
    counts, edges = np.histogram(data, bins=numBins)
    mean = np.mean(data)
    std = np.std(data)

    midpoints = (edges[1:] + edges[:-1]) / 2

    lengthBin = edges[2]-edges[1]
    scaleFactor = len(data) * lengthBin
    hist = counts/scaleFactor


    fig = plt.figure()

    topTitle = ''
    if name is not None:
        topTitle += name
    if title is not None or title_type is not None:
        if topTitle:
            topTitle += ' --'
    if title is not None:
        topTitle += ' '+title
    if title_type is not None:
        topTitle += ' '+title_type

    fig.suptitle(topTitle)

    histMax = np.max(hist)

    plt.plot(midpoints, hist, label='Scaled Histogram')
    plt.plot([mean]*2,     [0, 0.90*histMax], 'c-.', label='Mean')
    plt.plot([mean+std]*2, [0, 0.75*histMax], 'k:', label='+/- Stdev')
    plt.plot([mean-std]*2, [0, 0.75*histMax], 'k:')
    plt.plot(midpoints, norm.pdf(midpoints, mean, std), 'r', label='Gaussian')
    lab = 'Errors'
    if units is not None:
        lab += ' (%s)' % units
    plt.xlabel(lab)
    plt.ylabel('Counts')
    plt.legend()
    if saveFile is not None:
        plt.savefig(saveFile)

