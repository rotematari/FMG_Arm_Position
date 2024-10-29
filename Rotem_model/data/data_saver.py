from os import listdir
from os.path import isfile, join
import os 
# Get the current directory of the script being run
current_directory = os.path.dirname(os.path.realpath(__file__))

# Navigate up  directori
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# Change the working directory
os.chdir(parent_directory)

import numpy as np
import serial
import time as t

from NatNetClient import NatNetClient

import matplotlib.pyplot as plt 
import yaml
import argparse
import pandas as pd

# from FMG_project.utils.utils import print_not_numeric_vals
data_dir = r'data/new'

with open(r'config.yaml', 'r') as f:
    args = yaml.safe_load(f)

config = argparse.Namespace(**args)
# dirs = [f for f in listdir(dirpath)]
# make sure the 'COM#' is set according the Windows Device Manager /dev/ttyACM0
ser = serial.Serial('/dev/ttyACM0', 115200)

def is_not_numeric(x):
    try:
        float(x)
        return False
    except ValueError:
        return True
    
    print(non_numeric_values)

    return non_numeric_values


def write_first_line(f,config):
    sensor_headers = ','.join(config.fmg_index)
    label_headers = ','.join(config.label_index)
    f.write(sensor_headers+',')
    f.write(label_headers+',')
    f.write(config.session_time_stamp[0]+'\n')


def receiveNewFrame( frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
    print( "Received frame", frameNumber )

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame( id, position, rotation ):
    print( "Received frame for rigid body", id )

def receiveRigidBodyList( rigidBodyList, stamp ):
    for (ac_id, pos, quat, valid) in rigidBodyList:
        # print("rigidBodyList")
        # print(rigidBodyList)
        # print(type(rigidBodyList))
        # print(len(rigidBodyList))
        if not valid:
            # skip if rigid body is not valid
            continue
        
        # print('id: ', ac_id, 'pos:', pos, 'quat:', quat) 


def init_natnetClient():

    # start natnet interface
    natnet = NatNetClient(rigidBodyListListener=receiveRigidBodyList,server="132.66.51.232")#rigidBodyListListener=receiveRigidBodyList)

    keys = ['chest', 'shoulder', 'elbow', 'wrist']
    chest = 1
    shoulder = 2
    elbow = 3
    wrist = 4
    return natnet

    # This dictionary matches the rigid body id (key) to it's name (value)
    motive_matcher = {chest: 'chest',
                        shoulder: 'shoulder',
                        shoulder: 'elbow',
                        elbow: 'wrist',}


def write_line(f,marker_data,sesion_time_stamp,motive_matcher):

    locations = {
        'chest':[],
        'shoulder':[],
        'elbow':[],
        'wrist':[],
    }
    # ser.reset_input_buffer()
    line = ser.readline().decode("utf-8").rstrip(',\r\n') # Read a line from the serial port
    
    # sensor_string = line.split(',')
    
    # sensor_string = [line[i] for i in config.sensor_location ]
    # sensor_string = ','.join(sensor_string)
    sensor_string = line
    
    for j in range(len(marker_data)):
        locations[motive_matcher[marker_data[j][0]]].append(marker_data[j][1])
    # read serial line
    locations_string = ','.join(map(str,locations['shoulder'][0]+locations['elbow'][0]+locations['wrist'][0]))
    if len(sensor_string.split(',')) == 33:
        # sensor_string , marker_string , sesion_time_stamp
        f.write(f'{sensor_string}' + ',' + f'{locations_string}' +','+ f'{sesion_time_stamp}' + '\n')


    # return sesion_time_stamp
def print_not_numeric_vals(df,config):

    mask = df[config.fmg_index+config.label_index].applymap(is_not_numeric)
    non_numeric_values = df[mask].stack().dropna()
    print(non_numeric_values)

    return non_numeric_values

def plot_data(config,data: pd.DataFrame):

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,5))
    data = data.reset_index(drop=True)
    ax1.plot(data[config.label_index])
    ax2.plot(data[config.fmg_index])
    ax1.legend()

    plt.show() 





if __name__ == '__main__':
    for i in range(10):
        ser.readline()
    
    print("make sure the assets set to xyz")
    keys = ['chest', 'shoulder', 'elbow', 'wrist']
    chest = 1
    shoulder = 2
    elbow = 3
    wrist = 4
    motive_matcher = {chest: 'chest',
                        shoulder: 'shoulder',
                        elbow: 'elbow',
                        wrist: 'wrist',
                    }    
    sesion_time_stamp = t.strftime("%d_%b_%Y_%H_%M", t.gmtime())
    file_name = sesion_time_stamp + '_test_time'+'.csv'
    NatNet = init_natnetClient()
    print(file_name)
    f = open(join(data_dir, file_name), "w")
    
    write_first_line(f,config=config)
    NatNet.run()
    t.sleep(10)
    marker_data = NatNet.rigidBodyList
    t_start = t.time()
    num_of_sampls = 5000
    for i in range(num_of_sampls):
        
        marker_data = NatNet.rigidBodyList
        write_line(f,sesion_time_stamp=sesion_time_stamp ,marker_data=marker_data,motive_matcher = motive_matcher)

        if i%100==0:
            print(i)
    
    f.close()
    NatNet.stop()

    t_end = t.time()
    print(f'time it took for {num_of_sampls} = {t_end-t_start} sec')
    
    ser.close()
    print("finished")

    # checks data 
    df = pd.read_csv(join(data_dir,file_name))
    plot_data(config=config,data=df)

    not_numeric_vals = print_not_numeric_vals(df,config=config)

    if not_numeric_vals.shape[0] == 0:
        plot_data(config=config,data=df)

    else :
        print("clean data")