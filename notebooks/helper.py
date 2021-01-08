'''
 # @ Author: Saeed Arabi
 # @ Create Time: 2020-12-29 16:24:57
 # @ Email: arabi@iastate.edu
 '''

import pandas as pd
import datetime
import numpy as np



def dist_cal(x):
    # calculate distance base on heigth of bbox(x)
    return 851 * 1.6825 / x


def prepare_df(filename, print_log=False):
    # prepare dataset for each detection
    if print_log:
        print('Loading file {}'.format(filename))
    df = pd.read_csv(filename)
    df['dist'] = df.H.apply(dist_cal)
    df = df[df.class_ID == 0]
    df['C_x'] = df.L + .5*df.W
    df['C_y'] = df['T'] + .5*df.H
    df['center'] = (df.C_x**2 + df.C_y**2)**.5
    return df


def find_ID_mode(df):
    return df[df.obj_ID == df.obj_ID.mode()[0]]


def get_first_and_last_value(df, col):
    first = df.iloc[0][col]
    last = df.iloc[-1][col]
    return first, last


def cluster(filename, n_last_elements=1000):
    df = prepare_df(filename)
    df = df.tail(n_last_elements)
    unique_ID = df.obj_ID.unique()
    for i in range(len(unique_ID)):
        df_mode = find_ID_mode(df)
        first_center_mode, last_center_mode = get_first_and_last_value(
            df_mode, 'center')
        first_frame_mode, last_frame_mode = get_first_and_last_value(
            df_mode, 'frame_num')
        l = []
        groups = df.groupby('obj_ID')
        for name, group in groups:
            if not name == df.obj_ID.mode()[0]:
                first_center, last_center = get_first_and_last_value(
                    group, 'center')
                first_frame, last_frame = get_first_and_last_value(
                    group, 'frame_num')
                if ((abs(first_center_mode - last_center) < 20) and (abs(first_frame_mode - last_frame) < 20)
                        or (abs(last_center_mode - first_center) < 20) and (abs(last_frame_mode - first_frame) < 20)):
                    l.append(name)
            else:
                l.append(name)
        df.loc[df.obj_ID.isin(l), 'obj_ID'] = min(l)
    df.loc[~df.obj_ID.isin(l), 'obj_ID'] = -1
    df.loc[df.obj_ID.isin(l), 'obj_ID'] = 1
    return df


def prepare_detections_for_model(file, df_raw_speed):
    df_detection = cluster(file)
    df_detection['datetime_time'] = [i.round('1s').time() for i in pd.date_range(min(df_raw_speed.datetime), max(df_raw_speed.datetime), len(df_detection))]    
    return df_detection

def prepare_speed_profiles_for_model(file):          
    df_raw_speed = pd.read_csv(file)
    df_raw_speed.columns = ['ST_unit', 'Latitude', 'Longitude', 'Date', 'Time_UTC', 'newDateTime', 'Speed_knots', 'Speed_MPH', 'Speed_mps', 'valid']
    df_raw_speed['datetime'] = [datetime.datetime.strptime(str(i), '%H:%M:%S') for i in df_raw_speed.Time_UTC]
    df_raw_speed['datetime_time'] = df_raw_speed.datetime.dt.time
    return df_raw_speed

def merge_speed_and_detection(df_detection, df_raw_speed, annotations, filename):
    df_detection = pd.merge(df_detection,df_raw_speed[['datetime_time','Speed_MPH']],on='datetime_time', how='left')
    df_detection.Speed_MPH.interpolate(inplace = True)
    df_detection = df_detection[df_detection.obj_ID == 1]
    df_detection['filename'] = filename
    df_detection = pd.merge(df_detection,annotations[[
        'filename','Type of vehicle','Type of road']], on='filename', how='left')
    df_detection = df_detection[['frame_num', 'dist', 'center', 'datetime_time',
                                'Speed_MPH', 'filename', 'Type of vehicle', 'Type of road']]
    return df_detection

def smooth_scatter(s,window):
    y = s
    N = window
    if len(s)>0:
        y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
        s_smooth = pd.Series(np.convolve(y_padded, np.ones((N,))/N, mode='valid'), index=s.index)                 
    else:
        print("len(series)=<0")
    return s_smooth
