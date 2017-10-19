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

matplotlib.style.use('ggplot')
params = {'legend.fontsize': 5,
         'axes.labelsize': 5,
         'axes.titlesize': 5,
         'xtick.labelsize': 5,
         'ytick.labelsize': 5}
matplotlib.pylab.rcParams.update(params)
#RECORD_TIME_S = 60*5
RECORD_TIME_S = 60*2
DATA_SIZE   = (2*3*2)
MPU_PERIOD  = 1/200 
while True:
    sers = []
    sers.append(serial.Serial(port='/dev/ttyUSB0', baudrate=115200))
    #sers.append(serial.Serial(port='/dev/ttyUSB1', baudrate=115200))
    datas = []
    for ser in sers:
        datas.append([])
        ser.flush()
    
    for idx,ser in enumerate(sers):
        ser.read(1)
    
    startdate = datetime.datetime.now()
    starttime = time.time()
    last = -1
    while(time.time() - starttime) < RECORD_TIME_S:
        for idx,ser in enumerate(sers):
            datas[idx].append(ser.read(1))
        if int(time.time() - starttime) != last:
            sys.stdout.write('\r' + str(int(time.time() - starttime)) + 's')
            sys.stdout.flush()
        last = int(time.time() - starttime)
    for ser in sers:
        ser.close()
    parsedDatas = []
    for data in datas:
        finaldata = []
        lastidx = -1
        for idx in range(len(data)-1):
            if (ord(data[idx]) != 73 and ord(data[idx+1]) == 74) or (idx == 0 and ord(data[idx+1]) == 74):
                if lastidx != -1:
                    finaldata.append(data[lastidx:idx+1])
                lastidx = idx + 2
        data = finaldata

        data = [b''.join(d).replace(bytes([73, 73]), bytes([73])).replace(bytes([73, 74]), bytes([74])) for d in data]
        last_mpu_timestamp = 0
        last_mpu_cntr = 0

        adc_datas = []
        adc_timestamps = []
        mpu_datas = []
        mpu_timestamps = []
        for d in data:
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
                        mpu_timestamps.append((mpu_cntr + i/DATA_SIZE)*(MPU_PERIOD)*1000000 + last_mpu_timestamp)
            elif d[0] == 1:
                if len(d) >= 5:
                    last_mpu_timestamp = struct.unpack('<I', d[1:5])[0]
            elif d[0] == 2:
                if len(d) >= 7:
                    if last_mpu_timestamp == 0:
                        last_mpu_timestamp = struct.unpack('<I', d[1:5])[0]
                    adc_timestamps.append(struct.unpack('<I', d[1:5])[0])
                    adc_datas.append(struct.unpack('<H', d[5:7])[0])
            elif d[0] == 3:
                if len(d) >= 5:
                    error_timestamp = struct.unpack('<I', d[1:5])[0]
                    print(error_timestamp)
        start_timestamps = min([min(adc_timestamps), min(mpu_timestamps)])
        #adc_timestamps = [datetime.datetime.utcfromtimestamp(tst - start_timestamps) for tst in adc_timestamps]
        #mpu_timestamps = [datetime.datetime.utcfromtimestamp(tst - start_timestamps) for tst in mpu_timestamps]
        adc_timestamps = [startdate + datetime.timedelta(microseconds = (tst - start_timestamps)) for tst in adc_timestamps]
        mpu_timestamps = [startdate + datetime.timedelta(microseconds = (tst - start_timestamps)) for tst in mpu_timestamps]

        adc = pd.Series(data=adc_datas,index = adc_timestamps) 
        mpu = pd.DataFrame(data=mpu_datas, index = mpu_timestamps)

        mpu.to_hdf('mpu' + str(int(time.time())) + '.h5','data', format='table', append=False)
        adc.to_hdf('adc' + str(int(time.time())) + '.h5','data', format='table', append=False)
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(adc)
        plt.subplot(2,1,2)
        plt.plot(mpu)
        plt.show()

    continue
    df = pd.DataFrame(data={'rssi':parsedDatas[0][2], 'ch': parsedDatas[0][1]}, index = parsedDatas[0][0])
    title = 'data_' + str(time.time()).split('.')[0] + '.h5'
    df.to_hdf(title,'data', format='table', append=True)
    print(title)
    plt.figure()
    for data in parsedDatas:
        data[0] -= data[0][0]
        #data[2][data[2] < -50] = np.nan
        for ch in range(0,17):
            print("*"*20)
            print(ch)
            chdata = data[2].copy()
            chdata[data[1] != ch] = np.nan
            plt.plot(data[0], chdata, '.')
            plt.plot(data[0][data[1] == ch], chdata[data[1] == ch], '-')
            continue
            idx = 0
            vals = []
            lastidx = 0
            while idx < len(chdata):
                if (len(vals) > 0) and (np.isnan(chdata[idx])):
                    chdata[lastidx:idx] = np.median(vals)
                    lastidx = idx
                    vals = []
                elif not np.isnan(chdata[idx]):
                    vals.append(chdata[idx])
                idx += 1
    plt.show()
    continue
    starttime = 0
    split = 50
    TSTEP = 5000000/split
    histarray = []
    fullarray = []
    for i in range(split):
        name = str(i)
        while len(name) < 3:
            name = '0' + name
        #plt.figure()
        histlist = []
        datamedian = []
        for data in parsedDatas:
            #plt.title(name)
            #plt.hist(data[1][(~np.isnan(data[1]))*(data[0] < starttime+TSTEP)*(data[0] > starttime)], 50, range = [-60, -20])
            histlist.append(np.histogram(data[1][(~np.isnan(data[1]))*(data[0] < starttime+TSTEP)*(data[0] > starttime)], 40, range = [-60, -20])[0])
            datamedian.append(np.median(data[1][(~np.isnan(data[1]))*(data[0] < starttime+TSTEP)*(data[0] > starttime)]))
        fullarray.append(datamedian)
        #plt.savefig('imgs/' + name, dpi=200)
        #plt.close()
        histarray.append(histlist[0])
        print(name)
        starttime += TSTEP
    plt.figure()
    plt.plot(histarray)
    plt.figure()
    plt.plot(fullarray)
    plt.show()
    input()
