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
params = {'legend.fontsize': 5,
         'axes.labelsize': 5,
         'axes.titlesize': 5,
         'xtick.labelsize': 5,
         'ytick.labelsize': 5}
matplotlib.pylab.rcParams.update(params)

def detectHB(datas):
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
    return [x,y]

def splitHB(x, datas):
    PLOT_PARTIAL = False
    lastX = x[0]
    firstX = x[0]
    splitData = []
    final = []
    targetcolumn = 'acc_x'
    N = 0
    for index in x[1:]:
        data = datas[(datas.index < index) & (datas.index >= lastX)]
        first = data[pd.notnull(datas['acc_x'])].index[0]
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
        lastX = index
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
    plt.subplot(2,1,2)
    for c_cntr, mpu in enumerate(mpus[3:]):
        valids = pd.notnull(final["final_"+mpu])
        data = final["final_"+mpu][valids] 
        data -= data.mean()
        plt.plot(data.index, data, c[c_cntr] + '-')
    plt.title("final")

files = [['_test04_16_10_2017.h5'], ['_test03_15_10_2017.h5'], ['_test02_15_10_2017.h5'], ['_test01_15_10_2017.h5']]
basefile = 'data/'
baseimg  = 'plot/'
SPLIT_N = 12
for f in files:
    if len(f) == 1:
        f = [basefile + 'adc' + f[0], basefile + 'mpu' + f[0]]
    hb = pd.read_hdf(f[0],'data').rename('hb')
    figname = f[0].split('adc')[1].split('.')[0]
    if figname[0] == '_':
        figname = figname[1:]
    figname = baseimg + figname
    print(figname)
    hb = hb - hb.mean()
    mpu = pd.read_hdf(f[1],'data').rename(columns={0:'acc_x', 1:'acc_y', 2:'acc_z', 3:'gy_x', 4:'gy_y', 5:'gy_z'})
    full_data = pd.concat([mpu, hb.rename('hb')], axis=1, names=[0,1,2,3,4,5,6]).resample('1L').mean().interpolate()
    step = (full_data.index[-1] - full_data.index[0])/SPLIT_N
    start = full_data.index[0]
    for i in range(SPLIT_N):
        data = full_data[(full_data.index >= start) & (full_data.index < start + step)]
        start += step
        [x,y] = detectHB(data)
        plt.savefig(figname + '_hb_' + str(i))
        splitHB(x, data)
        plt.savefig(figname + '_final_mean_' + str(SPLIT_N) + '_' + str(i))
    plt.close('all')
    break
    continue
    [dfile, hbfile] = f
    hbraw = open(hbfile, 'r')
    hbraw = hbraw.read()
    print(hbraw)
    fjdla
    data = pd.read_hdf(f, 'data')
    timestamps = []
    datas = []
    data.index = data.index/1000000
    data.index = data.index - data.index[0]
    data.index = pd.to_datetime(data.index, unit='s')
    for ch in range(17):
        datas.append(data[data['ch'] == ch].rename(columns = {'rssi': 'ch'+str(ch)}))
    datas = pd.concat(datas).sort_index()
    datas = datas.drop("ch",1)
    
    #plt.figure()
    #plt.plot(datas.index, datas['ch0'], 'r.')
    #plt.plot(datas.index[~np.isnan(datas['ch0'])], datas['ch0'][~np.isnan(datas['ch0'])], 'r-')
    #plt.plot(datas.index, datas['ch1'], 'b.')
    #plt.plot(datas.index[~np.isnan(datas['ch1'])], datas['ch1'][~np.isnan(datas['ch1'])], 'b-')
    #plt.legend()
    
    datas = datas.resample('1L').mean()
    step_size = 100
    validcntr = datas.rolling(window=step_size, center=False, min_periods=1).count()
    datas = datas.rolling(window=step_size, center=False, min_periods=1).sum()/validcntr
    datasstd = datas.rolling(window=step_size, center=False, min_periods=1).std()
    
    [maxpeaks, minpeaks] = peakdetect.peakdetect(datas['ch0'][~np.isnan(datas['ch0'])], lookahead=200)
    ch0_index = list(datas.index[~np.isnan(datas['ch0'])])
    x, y = zip(*maxpeaks)
    remove_idx = (np.asarray([0] + list(np.diff(y))) > -20)
    y = np.asarray(y)
    x = np.asarray(x)
    y = y[remove_idx]
    x = x[remove_idx]
    #y = y[::2]
    #x = x[::2]
    peak_x = [val for i,val in enumerate(ch0_index) if i in x]
    
    blocks = [datas.index[(datas.index >= peak_x[idx])*(datas.index < peak_x[idx+1])] for idx in range(len(peak_x)-1)]
    maxlen = max([len(block) for block in blocks])
    counter = np.asarray([0]*maxlen)
    datablocks = []
    counterblocks = []
    endarray = np.asarray([False]*maxlen)
    for block in blocks:
        basearray = np.asarray([0.0]*maxlen)
        counterarray = np.asarray([0.0]*maxlen)
        endarray[len(block)-1] = True
        basearray[0:len(block)] = basearray[0:len(block)] + datas['ch1'][block]
        counterarray[0:len(block)] = counterarray[0:len(block)] + ~np.isnan(basearray[0:len(block)])
        datablocks.append(basearray)
        counterblocks.append(counterarray)
    base_counter = np.sum(counterblocks, axis = 0)
    base_sample = np.nansum(datablocks, axis=0)/base_counter
    idxs = np.asarray(range(len(endarray)))

    plt.figure()
    plt.plot(base_sample)
    plt.plot(idxs[endarray], base_sample[endarray], 'kx')
    plt.title(f) 
    break
    continue
    
    plt.figure()
    
    plt.plot([datas.index[0], datas.index[-1]], [datas['ch0'].mean(), datas['ch0'].mean()], 'k-')
    plt.plot(peak_x, y, 'kx')
    
    plt.plot(datas.index[~np.isnan(datas['ch0'])], datas['ch0'][~np.isnan(datas['ch0'])], 'r.')
    plt.plot(datas.index[~np.isnan(datas['ch0'])], datas['ch0'][~np.isnan(datas['ch0'])], 'r-')
    
    plt.plot(datas.index[~np.isnan(datas['ch1'])], [0] + list(np.diff(datas['ch1'][~np.isnan(datas['ch1'])]) + 150), 'b-')
    plt.plot(datas.index[~np.isnan(datas['ch1'])], [0] + list(np.diff(datas['ch1'][~np.isnan(datas['ch1'])]) + 150), 'b.')
    
    #plt.plot(datas.index[~np.isnan(datas['ch1'])], datas['ch1'][~np.isnan(datas['ch1'])], 'b-')
    #plt.plot(datas.index[~np.isnan(datas['ch1'])], datas['ch1'][~np.isnan(datas['ch1'])], 'b.')
    
    plt.plot(datas.index[~np.isnan(datas['ch1'])], datasstd['ch1'][~np.isnan(datas['ch1'])]+150, 'g-')
    plt.plot(datas.index[~np.isnan(datas['ch1'])], datasstd['ch1'][~np.isnan(datas['ch1'])]+150, 'g.')
    plt.legend()
    plt.title(f) 
    continue
    plt.show()
    exit()
    fdjsklfas
    ch1s = np.asarray(np.where(data.ch == 1)[0])
    starts = ch1s[[0] + list(np.where(np.diff(ch1s)>1)[0] + 1)]
    avgs = []
    laststart = starts[0]
    for start in starts[1:]:
        avgs.append(data.rssi[laststart:start].mean())
        laststart = start
    avgs.append(data.rssi[laststart:].mean())
    print(avgs)
    avgs = np.asarray(avgs)
    print(avgs)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(data.index, data.rssi, 'r-')
    plt.plot(starts, data.rssi[starts], 'bx')
    plt.subplot(2,1,2)
    plt.plot(data.index, data.ch, 'r-')
    plt.plot(starts, data.ch[starts], 'bx')
    
    plt.figure()
    print(avgs)
    plt.plot(data.index[starts[1:]], np.diff(avgs),'r-')
    plt.plot(data.index[starts[1:]], np.diff(avgs),'r.')
    plt.show()
plt.show()
