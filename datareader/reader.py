import time
import serial
import struct
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates
import matplotlib.pylab
import pandas as pd
import numpy as np
import math
import datetime
import sys
import glob
import os
import threading

matplotlib.style.use('ggplot')
params = {'legend.fontsize': 5,
         'axes.labelsize': 5,
         'axes.titlesize': 5,
         'xtick.labelsize': 5,
         'ytick.labelsize': 5}
matplotlib.pylab.rcParams.update(params)
#RECORD_TIME_S = 60*5
RECORD_TIME_S = 60*10
DATA_SIZE   = (2*3*2)
MPU_PERIOD  = 1/200 


def readRawData(record_time_s, store_partial = False):
    sers = []
    ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
    datas = []
    partialdatas = []
    if store_partial:
        filename = 'rawdata/rawdata_' + str(int(time.time())).split('.')[0] + '.raw'
        raw_file = open(filename,'wb')
    ser.flush()
    
    for idx,ser in enumerate(sers):
        ser.read(1)
    
    starttime = time.time()
    last = -1
    print("Starting record of " + str(record_time_s) + "S")
    while(time.time() - starttime) < record_time_s:
        if store_partial:
            partialdatas.append(ser.read(1))
            if len(partialdatas) > 200:
                datas += partialdatas
                raw_file.write(b''.join(partialdatas))
                partialdatas = []
        else:
            datas.append(ser.read(1))
        if int(time.time() - starttime) != last:
            sys.stdout.write('\r' + str(int(time.time() - starttime)) + 's')
            sys.stdout.flush()
        last = int(time.time() - starttime)
    for idx, ser in enumerate(sers):
        if store_partial:
            datas += partialdatas
            raw_file.write(b''.join(partialdatas))
            partialdatas = []           
            raw_file.close()
        ser.close()
    return datas

def loadDatas(filename = None):
    if filename == None:
        list_of_files = glob.glob('rawdata/*.raw') # * means all if need specific format then *.csv
        filename = max(list_of_files, key=os.path.getctime)
    raw_file = open(filename, 'rb')
    datas = raw_file.read()
    datas = [bytes([d]) for d in datas]
    return datas

def parseDatas(datas, store=True):
    parsedDatas = []
    startdate = datetime.datetime.now()
    finaldatas = []
    lastidx = -1
    for idx in range(len(datas)-1):
        if (ord(datas[idx]) != 73 and ord(datas[idx+1]) == 74) or (idx == 0 and ord(datas[idx+1]) == 74):
            if lastidx != -1:
                finaldatas.append(datas[lastidx:idx+1])
            lastidx = idx + 2
    datas = finaldatas

    datas = [b''.join(d).replace(bytes([73, 73]), bytes([73])).replace(bytes([73, 74]), bytes([74])) for d in datas]
    last_mpu_timestamp = 0
    last_mpu_cntr = 0

    adc_datas = []
    adc_timestamps = []
    mpu_datas = []
    mpu_timestamps = []
    for d in datas:
        if len(d) == 0:
            continue
        if d[0] == 0:
            if last_mpu_timestamp > 0:
                if len(d) < 5:
                    continue
                mpu_cntr = math.ceil(struct.unpack('<I', d[1:5])[0]/DATA_SIZE) - last_mpu_cntr
                if last_mpu_cntr == 0:
                    last_mpu_cntr = mpu_cntr
                    mpu_cntr = 0
                d = d[5:]
                for i in range(0, len(d), DATA_SIZE):
                    if (len(d) - i) < DATA_SIZE:
                        continue
                    acc_x = struct.unpack(">h", d[i:i+2])[0]
                    acc_y = struct.unpack(">h", d[i+2:i+4])[0]
                    acc_z = struct.unpack(">h", d[i+4:i+6])[0]
                    gyr_x = struct.unpack(">h", d[i+6:i+8])[0]
                    gyr_y = struct.unpack(">h", d[i+8:i+10])[0]
                    gyr_z = struct.unpack(">h", d[i+10:i+12])[0]
                    mpu_datas.append([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z])
                    mpu_timestamp = (mpu_cntr + i/DATA_SIZE)*(MPU_PERIOD)*1000000 + last_mpu_timestamp
                    if len(mpu_timestamps) > 0:
                        if ((mpu_timestamp - mpu_timestamps[-1]) != 5000):
                            pass
                            #print("0: " + str(mpu_timestamp - mpu_timestamps[-1]) + '|' + str(mpu_cntr))
                    mpu_timestamps.append(mpu_timestamp)
        elif d[0] == 1:
            if len(d) >= 5:
                new_mpu_timestamp = struct.unpack('<I', d[1:5])[0]
                if new_mpu_timestamp > last_mpu_timestamp:
                    print("1: " + str(last_mpu_timestamp))
                    last_mpu_timestamp = new_mpu_timestamp
                    last_mpu_cntr = 0
                else:
                    print("01: " + str(last_mpu_timestamp))
        elif d[0] == 2:
            if len(d) >= 7:
                if last_mpu_timestamp == 0:
                    last_mpu_timestamp = struct.unpack('<I', d[1:5])[0]
                    print("1x: " + str(last_mpu_timestamp))
                adc_timestamps.append(struct.unpack('<I', d[1:5])[0])
                adc_datas.append(struct.unpack('<H', d[5:7])[0])
        elif d[0] == 3:
            if len(d) >= 5:
                error_timestamp = struct.unpack('<I', d[1:5])[0]
                print(error_timestamp)
    if len(mpu_timestamps) == 0:
        print("ERROR no mpu data")
        return None
    start_timestamps = min([min(adc_timestamps), min(mpu_timestamps)])
    #adc_timestamps = [datetime.datetime.utcfromtimestamp(tst - start_timestamps) for tst in adc_timestamps]
    #mpu_timestamps = [datetime.datetime.utcfromtimestamp(tst - start_timestamps) for tst in mpu_timestamps]
    adc_timestamps = [startdate + datetime.timedelta(microseconds = (tst - start_timestamps)) for tst in adc_timestamps]
    mpu_timestamps = [startdate + datetime.timedelta(microseconds = (tst - start_timestamps)) for tst in mpu_timestamps]

    adc = pd.Series(data=adc_datas,index = adc_timestamps) 
    mpu = pd.DataFrame(data=mpu_datas, index = mpu_timestamps)
    if store: 
        nametime = str(int(time.time()))
        mpu.to_hdf('data/mpu' + nametime + '.h5','data', format='table', append=False)
        adc.to_hdf('data/adc' + nametime + '.h5','data', format='table', append=False)
    return [mpu, adc]

def plotDatas(datas, timeout = None, zoomplot = 10):
    mpu = datas[0]
    adc = datas[1]
    fig = plt.figure()
    if timeout != None:
        timer = fig.canvas.new_timer(interval=3000)
        timer.add_callback(plt.close)
    plt.subplot(3,1,1)
    plt.plot(adc[adc.index > (adc.index.max() - datetime.timedelta(seconds=zoomplot))])
    plt.subplot(3,1,2)
    plt.plot(mpu[mpu.index > (mpu.index.max() - datetime.timedelta(seconds=zoomplot))])
    plt.subplot(3,1,3)
    plt.plot(mpu)
    if timeout != None:
        timer.start()
    plt.show()

def plotContinous(record_time_s):
    starttime = time.time()
    while(time.time() - starttime) < record_time_s:
        datas = loadDatas()
        datas = parseDatas(datas, store=False)
        if datas == None:
            time.sleep(1)
            continue
        plotDatas(datas, timeout = 3000)

        

t = threading.Thread(target=plotContinous, args=(RECORD_TIME_S,))
t.start()
datas = readRawData(RECORD_TIME_S, True)
datas = parseDatas(datas)
plotDatas(datas)
