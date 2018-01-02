import time
import struct
import sys
import math
import datetime
import random

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates
import matplotlib.pylab
import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal
import peakdetect
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use('ggplot')
size_things = 7
params = {'legend.fontsize':    size_things,
         'axes.labelsize':      size_things,
         'axes.titlesize':      size_things,
         'xtick.labelsize':     size_things,
         'ytick.labelsize':     size_things}
matplotlib.pylab.rcParams.update(params)

KALMAN_ON_FULL_SET  = True
USE_TRAPZ           = True
FIX_ROTATION        = True

STORE_IMG           = True

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

def plt_signal(datas, x, l0, l1):
    plt.figure()
    acc_x = datas['acc_x']- datas['acc_x'].mean() 
    acc_y = datas['acc_y']- datas['acc_y'].mean() 
    acc_z = datas['acc_z']- datas['acc_z'].mean() 
    plt.plot(datas.index, acc_x)
    plt.plot(datas.index, acc_y)
    plt.plot(datas.index, acc_z)

    central_hb = 5
    if l0 != None:
        if len(x) < (central_hb+l0):
            return
        plt.xlim(x[central_hb], x[central_hb+l0])
        central_hb += math.floor(l0/2)
        indexs = (datas.index > x[central_hb]) & (datas.index <= x[central_hb+l0])
        acc_x = datas['acc_x'][indexs]- datas['acc_x'][indexs].mean() 
        acc_y = datas['acc_y'][indexs]- datas['acc_y'][indexs].mean() 
        acc_z = datas['acc_z'][indexs]- datas['acc_z'][indexs].mean() 
    if l1 != None:
        if len(x) < (central_hb+l1):
            return
        plt.plot([x[central_hb], x[central_hb+l1]], [0,0], 'k-')
        plt.plot([x[central_hb], x[central_hb+l1]], [0,0], 'wo')

    min_point = min([acc_x.min(), acc_y.min(), acc_z.min()])
    max_point = max([acc_x.max(), acc_y.max(), acc_z.max()])
    for hb in x:
        plt.plot([hb, hb], [min_point, max_point], 'g-')
    plt.ylim([min_point, max_point])

def plotSomeBeats(N, splitData):
    plt.figure()
    done_plot = []
    max_timestamp = 0
    for n in range(N):
        plt.subplot(N, 1, n+1)
        x = int(random.random()*len(splitData))
        while x in done_plot:
            x = int(random.random()*len(splitData))
        done_plot.append(x)
        max_timestamp = max([max_timestamp, (splitData[x].index.max() - splitData[x].index.min()).microseconds])
        idxs = [(idx - splitData[x].index.min()).microseconds for idx in splitData[x].index]
        plt.plot(idxs, splitData[x]['acc_x'])
        plt.plot(idxs, splitData[x]['acc_y'])
        plt.plot(idxs, splitData[x]['acc_z'])
    #print(max_timestamp)
    for n in range(N):
        plt.subplot(N, 1, n+1)
        plt.xlim([0, max_timestamp])

def splitHB(x, datas, figname = ''):
    PLOT_PARTIAL = False
    lastX = x[0]
    firstX = x[0]
    splitData = []
    final = []

    plt_signal(datas, x, 30, 7)
    plt.savefig(figname + '_zoom0', dpi=300)
    plt.close()

    plt_signal(datas, x, 7, 1)
    plt.savefig(figname + '_zoom1', dpi=300)
    plt.close()

    plt_signal(datas, x, 1, None)
    plt.savefig(figname + '_zoom2', dpi=300)
    plt.close()
    
    #Data was resampled at 1000Hz
    welch = sp.signal.welch(datas['acc_x'].values, fs = 1000, array=0)
    print(datas['acc_x'])
    print(welch)
    fdjkls


    targetcolumn = 'acc_x'
    N = 0
    M = 1
    mean_accs = []
    mean_vels  = []
    max_dist   = []
    lengths = []
    valid_indexs = []
    print("Calculating each beat")
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
        if (timelen.microseconds < 1500000) and (len(data) < 1250):
            resAcc = data['acc_x'].pow(2) + data['acc_y'].pow(2) + data['acc_z'].pow(2)
            resAcc = resAcc.pow(1/2)
            if resAcc.sum() < 20000000:
                if not KALMAN_ON_FULL_SET: 
                    data = runKalmanOnSet(data, acc_fields, gy_fields, 1000, X, P)
                    euler_fields = ['e1','e2','e3']
                    data[acc_fields] = rotateAcc(np.asarray(data[acc_fields]), np.asarray(data[euler_fields]))
                [pos, vel] = calculatePos(data[acc_fields], 1/1000, plot = False)
                pos = np.asarray(pos)
                pos = pd.DataFrame(data=pos, index = data.index, columns = ['pos_x', 'pos_y', 'pos_z'])
                dist = np.sqrt(np.sum(np.power(pos, 2), axis=1))
                data = pd.concat([pos, data], axis = 1)
                vel = np.asarray(vel)
                vel = pd.DataFrame(data=vel, index = data.index, columns = ['vel_x', 'vel_y', 'vel_z'])
                resVel = np.sqrt(np.sum(np.power(vel, 2), axis=1))
                data = pd.concat([vel, data], axis = 1)
                splitData.append(data)
                lengths.append(len(data))
                mean_accs.append(resAcc.sum()/len(data))
                mean_vels.append(resVel.sum()/len(data))
                max_dist.append(dist.max())
                valid_indexs.append(index)
        lastX = index
    plotSomeBeats(6, splitData)
    if STORE_IMG:
        plt.savefig(figname + '_samples', dpi=300)
        plt.close()

    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(valid_indexs, np.asarray(mean_accs))
    plt.title('mean acc')
    plt.subplot(4,1,2)
    plt.plot(valid_indexs, np.asarray(mean_vels))
    plt.title('mean vel')
    plt.subplot(4,1,3)
    plt.plot(valid_indexs, np.asarray(max_dist))
    plt.title('max dist')
    plt.subplot(4,1,4)
    plt.title('lenght')
    plt.plot(valid_indexs, np.asarray(lengths))
    if figname != '':
        plt.savefig(figname + '_works')
        plt.close()
    finalList = {}
    for column in splitData[0].columns:
        finalList['final_'+column] = [0, 0]
    indexs = []
    print("Join them")
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
        acc_data = final["final_"+mpu][valids] 
        acc_data -= acc_data.mean()
        final["final_"+mpu][valids] = acc_data
        plt.plot(acc_data.index, acc_data, c[c_cntr] + '-')
    columns = ["final_"+mpu for mpu in mpus[:3]]
    acc_datas = np.asarray(final[columns])
    plt.title("accelerometer")
    plt.subplot(2,1,2)
    for c_cntr, mpu in enumerate(mpus[3:]):
        valids = pd.notnull(final["final_"+mpu])
        gy_data = final["final_"+mpu][valids] 
        gy_data -= gy_data.mean()
        plt.plot(gy_data.index, gy_data, c[c_cntr] + '-')
    plt.title("gyroscope")
    if figname != '':
        plt.savefig(figname + '_final_mean', dpi=300)
        plt.close()
    return [acc_datas, gy_data]

def convertToQuartenions(acc, gy):
    fz = acc[0];
    fy = acc[1];
    fx = acc[2];
    phi   = math.atan2(fy,fz);
    theta = math.atan2(-fx, math.sqrt(math.pow(fy, 2) + math.pow(fz, 2)));

    q1 =  math.cos(theta/2) * math.cos(phi/2);
    q2 =  math.sin(phi/2)   * math.cos(theta/2);
    q3 =  math.cos(phi/2)   * math.sin(theta/2);
    q4 = -math.sin(phi/2)   * math.sin(theta/2);
    accq = [q1,q2,q3,q4];

    gyq = [];
    gyq.append([0    , -gy[0], -gy[1],  gy[2]])
    gyq.append([gy[0],  0    ,  gy[2], -gy[1]])
    gyq.append([gy[1], -gy[2],  0    ,  gy[0]])
    gyq.append([gy[2],  gy[1], -gy[0],  0    ])

    accq = np.asarray(accq)
    gyq  = np.asarray(gyq)

    return accq, gyq

def runKalman(accDataQ, gyDataQ,dt,oldX,oldP,Q,H,R):
    #Q: Driving noise variance matrix
    #H: Observation matrix
    #R: (C) noise variance matrix
    A = np.identity(4) + gyDataQ*dt/2;
    x = np.dot(A, oldX);
    P = np.dot(np.dot(A, oldP), np.transpose(A)) + Q;
    K = np.dot(np.dot(P, np.transpose(H)), (np.linalg.inv(np.dot(np.dot(H, P), np.transpose(H)) + R)))
    x = x + np.dot(K, (accDataQ - np.dot(H, x)));
    P = P - np.dot(np.dot(K, H), P);
    return x, P

def quartenionsToEuler(q):
    #This assumes that yaw is 0
    theta =  math.atan2(2*(q[0]*q[1] + q[2]*q[3]),1 - 2*(math.pow(q[1],2) + math.pow(q[2],2)))
    val = 2*(q[0]*q[2] - q[3]*q[1])
    if(abs(val) > 1):
        print(val)
        val = val/abs(val)
    PHI   =  math.asin(val)
    phi   =  math.atan2(2*(q[0]*q[3] + q[1]*q[2]),1 - 2*(math.pow(q[2],2) + math.pow(q[3],2)))
    euler = [np.real(theta), np.real(PHI), np.real(phi)]
    return euler

def eulertToQuartenions(x):
    q1 = [math.cos(x[0]/2), 0               , 0               , math.sin(x[0]/2)]
    q2 = [math.cos(x[1]/2), 0               , math.sin(x[1]/2), 0               ]
    q3 = [math.cos(x[2]/2), math.sin(x[2]/2), 0               , 0               ]
    q = np.dot(np.dot(q1, q2), q3)
    return q

def runKalmanOnSet(full_data, accFields,gyFields,sFreq,x0,P):
    #This function calculates the breathing rate from accelerometer and gyroscope data 
    #INPUT:
    #   accData: accelerometer data as a N x 3 matrix
    #   gyData:  gyroscope data as a N x 3 matrix
    #   sFreq:   sampling frequency (Hz)
    #   gain:    the gain applied to the max prominense to define the threashold 
    #   doPlot:  1 - plot or 0 - do not plot
    #OUTPUT:
    #   numberBR     The final number of breathin rates calculated with each direction as a 1x3 array
    #   result:      Nx3 data resultant from the complete filtering process
    #   resultPeaks: The peaks of that data
    #   Euler:       The resulting Euler angles from the kalman filter

    H = np.identity(4);
    Q = np.identity(4)*0.001;
    R = np.identity(4)*0.001;
    acc_datas = []
    gy_datas  = [] 
    for field in accFields:
        acc_datas.append(np.asarray(full_data[field]))
    for field in gyFields:
        gy_datas.append(np.asarray(full_data[field]))
    # Find the first valid value
    start_idx = None
    for idx, acc_x in enumerate(acc_datas[0]):
        valid = True
        for datas in [acc_datas, gy_datas]:
            for i in range(3):
                if np.isnan(datas[i][idx]):
                    valid = False
                    break
        if valid:
            start_idx = idx
            for datas in [acc_datas, gy_datas]:
                for i in range(3):
                    datas[i] = datas[i][idx:]
            break
    x = eulertToQuartenions(x0)
    euler = [];
    X = []
    X.append(x)
    print("Calculating Kalman:")
    for idx, acc_x in enumerate(acc_datas[0]):
        if int(1000*(idx+1)/len(acc_datas[0]))%10 == 0:
            sys.stdout.write(str(int(100*idx/len(acc_datas[0]))) + '%                       \r')
            sys.stdout.flush()
        acc_data = [data[idx] for data in acc_datas]
        gy_data  = [data[idx] for data in  gy_datas]
        accDataQ, gyDataQ = convertToQuartenions(acc_data, gy_data)
        x, P = runKalman(accDataQ,gyDataQ,1/sFreq,x,P,Q,H,R)
        X.append(x)
        #DEVERIA ESTART GUARDANDO O QUARTENION E CONVERTENDO DIRETO PARA ROTACAO
        euler.append(quartenionsToEuler(x))
    print("100%")
    euler = np.asarray(euler)
    euler = pd.DataFrame(data=euler, index = full_data.index[start_idx:], columns = ['e1', 'e2', 'e3'])
    data = pd.concat([euler, full_data], axis = 1)
    return data


def rotateAcc(accs, eulers):
    if not FIX_ROTATION:
        return accs
    for idx, euler in enumerate(eulers):
        valid = True
        for axis in accs[idx]:
            if np.isnan(axis):
                valid = False
                break
        if valid:
            R = [[math.cos(euler[1])   , -math.cos(euler[2])*math.sin(euler[1])             , math.sin(euler[1])*math.sin(euler[2])            ],
                 [math.cos(euler[0])*math.sin(euler[1]), math.cos(euler[0])*math.cos(euler[1])*math.cos(euler[2]) - math.sin(euler[0])*math.sin(euler[2])   , -math.cos(euler[2])*math.sin(euler[0]) - math.cos(euler[0])*math.cos(euler[1])*math.sin(euler[2])],
                 [math.sin(euler[0])*math.sin(euler[1]), math.cos(euler[0])*math.sin(euler[2])    + math.cos(euler[1])*math.cos(euler[2])*math.sin(euler[0]), math.cos(euler[0])*math.cos(euler[2])  - math.cos(euler[1])*math.sin(euler[0])*math.sin(euler[2]) ]]
            accs[idx] = np.dot(accs[idx], R)
    return accs

def calculatePos(accs, per, figname = '', plot = True):

    #accs[np.isfinite(accs)] -= np.mean(accs[np.isfinite(accs)], axis=0)
    #accs -= np.nanmean(accs, axis=0)
    accs = np.nan_to_num(accs)
    accs[abs(accs) < 1e-16] = 0

    pos = [0, 0, 0]
    timestamps = np.asarray(range(len(accs)))
    timestamps = per*timestamps
    if USE_TRAPZ:
        vels = sp.integrate.cumtrapz(accs, x=timestamps, axis=0)
        vels = np.append(vels, [[0, 0, 0]], axis = 0)
        vels[abs(vels) < 1e-16] = 0
        poss = sp.integrate.cumtrapz(vels, x=timestamps, axis=0)
        poss = np.append(poss, [[0, 0, 0]], axis = 0)
    else:
        vels = np.cumsum(accs, axis=0)/1000
        poss = np.cumsum(vels, axis=0)/1000

    pos1 = [pos[0] for pos in poss]
    pos2 = [pos[1] for pos in poss]
    pos3 = [pos[2] for pos in poss]
    if plot:
        plt.figure()
        plt.plot(vels)
        plt.title("vels")
        if figname != '':
            plt.savefig(figname + '_final_vels', dpi=300)
            plt.close()

        plt.figure()
        plt.plot(poss)
        plt.title("pos")
        if figname != '':
            plt.savefig(figname + '_final_pos', dpi=300)
            plt.close()
                
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(pos1, pos2, pos3, 'b-')
        ax.plot([pos1[ 0]], [pos2[ 0]], [pos3[ 0]], 'ro')
        ax.plot([pos1[-1]], [pos2[-1]], [pos3[-1]], 'rx')
        lim = 0.001
        max_lim = max([max(pos1), max(pos2), max(pos3)])
        min_lim = min([min(pos1), min(pos2), min(pos3)])
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        ax.set_zlim(min_lim, max_lim)
        plt.title("3d pos")
        if figname != '':
            plt.savefig(figname + '_final_pos_3d', dpi=300)
            plt.close()
    #for acc in accs:
    #    ax.plot([0, acc[0]], [0,acc[1]], [0,acc[2]], 'b->')
    #    acc = np.dot(acc, R)
    #    print(acc)
    #    ax.plot([0, acc[0]], [0,acc[1]], [0,acc[2]], 'r->')
    return [poss, vels]


files = [['_test06_19_10_2017.h5'], ['_test05_19_10_2017.h5'], ['_test04_16_10_2017.h5'], ['_test03_15_10_2017.h5'], ['_test02_15_10_2017.h5'], ['_test01_15_10_2017.h5']]
basefile = 'data/'
baseimg  = 'plot/'
SPLIT_NS = [1, 5, 10]
if not 'full_result_keeper' in locals():
    print("First RUN")
    full_result_keeper = {}
for SPLIT_N in SPLIT_NS:
    for f in files:
        if not f[0] in full_result_keeper:
            full_result_keeper[f[0]] = {}
        if not SPLIT_N in full_result_keeper[f[0]]:
            full_result_keeper[f[0]][SPLIT_N] = {}
        keeper = full_result_keeper[f[0]][SPLIT_N]
        
        if not 'files' in keeper:
            if len(f) == 1:
                f = [basefile + 'adc' + f[0], basefile + 'mpu' + f[0]]
            keeper['files'] = f
        f = keeper['files']

        #------LOAD AND FIX HB DATA--------------------
        figname = f[0].split('adc')[1].split('.')[0]
        if figname[0] == '_':
            figname = figname[1:]
        figname = baseimg + figname
        print('-'*100)
        print(figname)

        if not 'hb' in keeper:
            hb = pd.read_hdf(f[0],'data').rename('hb')
            hb = hb - hb.mean()
            keeper['hb'] = hb
        hb = keeper['hb']
        if not 'full_data' in keeper:
            mpu = pd.read_hdf(f[1],'data').rename(columns={0:'acc_z', 1:'gy_x', 2:'gy_y', 3:'gy_z', 4:'acc_x', 5:'acc_y'})
            constant_acc = 16384
            constant_gy = 131
            acc_fields = ['acc_x', 'acc_y', 'acc_z']
            gy_fields = ['gy_x', 'gy_y', 'gy_z']
            for acc_field in acc_fields:
                mpu[acc_field] /= constant_acc
            for gy_field in gy_fields:
                mpu[gy_field] /= constant_gy
            #---------JOIN AND RESAMPLE---------------
            full_data = pd.concat([mpu, hb.rename('hb')], axis=1, names=[0, 1, 2, 3, 4, 5, 6]).resample('1L').mean().interpolate()
            keeper['full_data'] = full_data
        full_data = keeper['full_data']

        step = (full_data.index[-1] - full_data.index[0])/SPLIT_N
        start = full_data.index[0]

        for i in range(SPLIT_N):
            if 'SPLIT_N' in keeper:
                if keeper['SPLIT_N'] != SPLIT_N:
                    keeper['runs'] = {}
                    keeper['SPLIT_N'] = SPLIT_N
            if not 'runs' in keeper:
                keeper['runs'] = {}
            if not i in keeper['runs']:
                keeper['runs'][i] = {}
            run_keeper = keeper['runs'][i]
            #------------SPLIT DATA AND ANALYZE
            print(str(i+1) + "/" + str(SPLIT_N))

            if STORE_IMG:
                name_file = figname
                if not KALMAN_ON_FULL_SET:
                    name_file += '_multikalman'
                if USE_TRAPZ:
                    name_file += '_trapz'
                name_file  += '_' + str(SPLIT_N) + '_' + str(i) + '_'
            else:
                name_file = ''
            if not 'data' in run_keeper:
                data = full_data[(full_data.index >= start) & (full_data.index < start + step)]
                if (pd.notnull(data[acc_fields + gy_fields]).sum().sum() == 0):
                    continue
                start += step
                P = np.identity(4)*0.01
                X = [0, 0, 0]
                if KALMAN_ON_FULL_SET: 
                    if not 'kalman' in keeper:
                        data = runKalmanOnSet(data, acc_fields, gy_fields, 1000, X, P)
                    euler_fields = ['e1','e2','e3']
                    data[acc_fields] = rotateAcc(np.asarray(data[acc_fields]), np.asarray(data[euler_fields]))
                run_keeper['data'] = data
            data = run_keeper['data']


            if not 'hb' in run_keeper:
                print("Calculating Heart Beats")
                [x,y] = detectHB(data, name_file)
                run_keeper['hb'] = [x,y]
            [x,y] = run_keeper['hb']
            if len(x) < 3:
                continue
            print("Calculating medium")
            [acc_datas, gy_data] = splitHB(x, data, name_file)
            calculatePos(acc_datas, 1/1000, figname = name_file)
            if not STORE_IMG:
                plt.show()
                plt.close('all')

