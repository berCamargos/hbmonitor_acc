import time
import struct
import sys
import math
import datetime

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates
import matplotlib.pylab
import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal
import peakdetect

matplotlib.style.use('ggplot')
size_things = 7
params = {'legend.fontsize':    size_things,
         'axes.labelsize':      size_things,
         'axes.titlesize':      size_things,
         'xtick.labelsize':     size_things,
         'ytick.labelsize':     size_things}
matplotlib.pylab.rcParams.update(params)

def detectHB(datas, figname = ''):
    rollingmean = datas['hb'].rolling('2S').mean()
    datas['hb'] -= rollingmean
    [maxpeaks, minpeaks] = peakdetect.peakdetect(datas['hb'][pd.notnull(datas['hb'])], x_axis = datas.index[pd.notnull(datas['hb'])])
    rollingmin  = datas['hb'].rolling('2S').min()
    rollingmax  = datas['hb'].rolling('2S').max()
    rollinggain = rollingmax - rollingmin
    rollinggain[rollinggain < 1] = 1
    datas['hb'] -= rollingmin
    datas['hb'] /= rollinggain
    x, y = zip(*maxpeaks)
    x = np.array(x)
    y = np.array(y)

    plt.figure()
    plt.plot(datas.index[pd.notnull(datas['hb'])], datas['hb'][pd.notnull(datas['hb'])], 'b-')
    plt.plot(x, datas['hb'][pd.notnull(datas['hb'])][x], 'ko')

    plt.plot(x[1:], np.diff(datas['hb'][pd.notnull(datas['hb'])][x]), 'ro')
    indexs = [True] + list(np.diff(datas['hb'][pd.notnull(datas['hb'])][x]) > -0.3)
    x = x[indexs]
    y = y[indexs]

    indexs = [True]*len(x)
    fastbeat = [False] + list(np.diff(x) < np.timedelta64(500*1000000, 'ns'))
    for idx, beat in enumerate(fastbeat):
        if beat == True:
            if (y[idx-1] > y[idx]):
                indexs[idx] = False
            else:
                indexs[idx-1] = False
    x = x[indexs]
    y = y[indexs]


    indexs = (datas['hb'][pd.notnull(datas['hb'])][x] > 0.5)
    y = y[indexs]
    x = x[indexs]

    plt.plot(x, datas['hb'][pd.notnull(datas['hb'])][x], 'rx')
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S.%f"))
    if figname != '':
        plt.savefig(figname + '_hb', dpi=300)
        plt.close()
    return [x,y]

def splitHB(x, datas, figname = ''):
    PLOT_PARTIAL = False
    lastX = x[0]
    firstX = x[0]
    splitData = []
    final = []
    targetcolumn = 'acc_x'
    N = 0
    works = []
    lengths = []
    valid_indexs = []
    for index in x[1:]:
        data = datas[(datas.index < index) & (datas.index >= lastX)]
        if len(data[pd.notnull(data['acc_x'])]) == 0:
            continue
        first = data[pd.notnull(data['acc_x'])].index[0]
        data = data[data.index >= first]
        data.index -= data.index[0]
        # Lets make all beats have exactelly 5s
        micro = data.index[-1].microseconds + data.index[-1].seconds*1000000
        gain = (5000000/(micro))
        #data.index *= gain
        data.index += firstX 
        #data = data.resample('5L').mean().bfill()
        data = data - data.mean()
        timelen = data.index.max() - data.index.min()
        if timelen.microseconds < 1500000:
            splitData.append(data)
            lengths.append(len(data))
            energy = data['acc_x'].pow(2) + data['acc_y'].pow(2) + data['acc_z'].pow(2)
            energy = energy.pow(1/2)
            works.append(energy.sum())
            valid_indexs.append(index)
        lastX = index
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(valid_indexs, np.asarray(works)/np.asarray(lengths))
    plt.title('mean work')
    plt.subplot(3,1,2)
    plt.plot(valid_indexs, np.asarray(works))
    plt.title('work')
    plt.subplot(3,1,3)
    plt.plot(valid_indexs, np.asarray(lengths))
    plt.title('lenght')
    if figname != '':
        plt.savefig(figname + '_works')
        plt.close()
    finalList = {}
    for column in splitData[0].columns:
        finalList['final_'+column] = [0, 0]
    indexs = []
    starts = [data.index[0] for data in splitData]
    ends = [data.index[-1] for data in splitData]
    indexs.append(min(starts) - datetime.timedelta(milliseconds = 500))
    indexs.append(max(ends) + datetime.timedelta(milliseconds = 500))
    final = pd.DataFrame(data=finalList, index=indexs).resample("1L").interpolate()
    for data in splitData:
        N += 1
        if N < 2:
            finalList = {}
            data.index += datetime.timedelta(milliseconds = 100)
            for column in data.columns:
                #final[column] = 
                finalcolumn = "final_"+column
                newfinal = pd.concat([data[column], final[finalcolumn]], axis=1).fillna(value=0).sum(axis=1)
                finalList["final_"+column] = newfinal
            final = pd.DataFrame(data=finalList, index=finalList["final_"+column].index)
            final -= final.mean()
        else:
            #In order to run a good crosscorrelation we should join them lets do it by padding with 0
            #data[targetcolumn] *= 0
            #data[targetcolumn] += 0.01
            #data[targetcolumn][(data.index >  (data.index[0] + datetime.timedelta(milliseconds = 10))) & (data.index <  (data.index[0] + datetime.timedelta(milliseconds = 110)))] += 1
            if PLOT_PARTIAL:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(final.index, final["final_"+targetcolumn]/(N-1), "r-")
                plt.plot(data.index, data[targetcolumn], "b-")
            if len(final["final_"+targetcolumn]) > len(data[targetcolumn]):
                corr = sp.signal.correlate(final["final_"+targetcolumn], data[targetcolumn])
                offset_ms = corr.argmax() - len(data[targetcolumn]) + 1 - ((data.index[0] - final.index[0]).microseconds/1000)
                offset_ms *= 1
                if offset_ms == 0:
                    subdir = 1
                else:
                    subdir = offset_ms/abs(offset_ms)
            else:
                corr = sp.signal.correlate(data[targetcolumn], final["final_"+targetcolumn])
                offset_ms = corr.argmax() - len(final["final_"+targetcolumn]) + 1
                offset_ms *= 1
                subdir = -1*(offset_ms/abs(offset_ms))
            data.index += subdir*datetime.timedelta(milliseconds = abs(int(offset_ms)))
            if PLOT_PARTIAL:
                plt.plot(data.index, data[targetcolumn], "k-")
                plt.subplot(2,1,2)
                plt.plot(corr)
                plt.show()
            finalList = {}
            for column in data.columns:
                #final[column] = 
                finalcolumn = "final_"+column
                newfinal = pd.concat([data[column], final[finalcolumn]], axis=1).fillna(value=0).sum(axis=1)
                finalList["final_"+column] = newfinal
            final = pd.DataFrame(data=finalList, index=finalList["final_"+column].index)
            final -= final.mean()
    N = 0
    mpus = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
    c = ['r','b','g','k','y','c']
    if N > 0:
        step = math.floor(len(splitData)/N)
        for idx in range(0, len(splitData), step):
            plt.figure()
            plt.subplot(2,1,1)
            for c_cntr, mpu in enumerate(mpus):
                valids = pd.notnull(splitData[idx][mpu])
                data = splitData[idx][mpu][valids]
                data -= data.mean()
                plt.plot(data.index, data, c[c_cntr] + '-')
            plt.title(str(idx))
            plt.subplot(2,1,2)
            valids = pd.notnull(splitData[idx]['hb'])
            data = splitData[idx]['hb'][valids] 
            plt.plot(data.index, data)
    plt.figure()
    plt.subplot(2,1,1)
    for c_cntr, mpu in enumerate(mpus[:3]):
        valids = pd.notnull(final["final_"+mpu])
        data = final["final_"+mpu][valids] 
        data -= data.mean()
        plt.plot(data.index, data, c[c_cntr] + '-')
    plt.title("accelerometer")
    plt.subplot(2,1,2)
    for c_cntr, mpu in enumerate(mpus[3:]):
        valids = pd.notnull(final["final_"+mpu])
        data = final["final_"+mpu][valids] 
        data -= data.mean()
        plt.plot(data.index, data, c[c_cntr] + '-')
    plt.title("gyroscope")
    if figname != '':
        plt.savefig(figname + '_final_mean', dpi=300)
        plt.close()

STORE_IMG = True
files = [['_test06_19_10_2017.h5'], ['_test05_19_10_2017.h5'], ['_test04_16_10_2017.h5'], ['_test03_15_10_2017.h5'], ['_test02_15_10_2017.h5'], ['_test01_15_10_2017.h5']]
basefile = 'data/'
baseimg  = 'plot/'
SPLIT_NS = [1, 5, 10]
for SPLIT_N in SPLIT_NS:
    for f in files:
        if len(f) == 1:
            f = [basefile + 'adc' + f[0], basefile + 'mpu' + f[0]]
        hb = pd.read_hdf(f[0],'data').rename('hb')
        figname = f[0].split('adc')[1].split('.')[0]
        if figname[0] == '_':
            figname = figname[1:]
        figname = baseimg + figname
        print('-'*100)
        print(figname)
        hb = hb - hb.mean()
        mpu = pd.read_hdf(f[1],'data').rename(columns={0:'acc_z', 1:'gy_x', 2:'gy_y', 3:'gy_z', 4:'acc_x', 5:'acc_y'})
        constant_acc = 16384
        constant_gy = 131
        acc_fields = ['acc_x', 'acc_y', 'acc_z']
        gy_fields = ['gy_x', 'gy_y', 'gy_z']
        for acc_field in acc_fields:
            mpu[acc_field] /= constant_acc
        for gy_field in gy_fields:
            mpu[gy_field] /= constant_gy
        full_data = pd.concat([mpu, hb.rename('hb')], axis=1, names=[0, 1, 2, 3, 4, 5, 6]).resample('1L').mean().interpolate()
        step = (full_data.index[-1] - full_data.index[0])/SPLIT_N
        start = full_data.index[0]
        for i in range(SPLIT_N):
            data = full_data[(full_data.index >= start) & (full_data.index < start + step)]
            start += step
            if STORE_IMG:
                name_file = figname + '_' + str(SPLIT_N) + '_' + str(i) + '_'
            else:
                name_file = ''
            [x,y] = detectHB(data, name_file)
            splitHB(x, data, name_file)
        if not STORE_IMG:
            plt.show()
            plt.close('all')

