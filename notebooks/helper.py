'''
 # @ Author: Saeed Arabi
 # @ Create Time: 2020-12-29 16:24:57
 # @ Email: arabi@iastate.edu
 '''

import pandas as pd


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
