'''
 # @ Author: Saeed Arabi
 # @ Create Time: 2020-12-29 16:24:57
 # @ Email: arabi@iastate.edu
 '''

import pandas as pd
import datetime
import numpy as np
import pickle
from scipy.stats import norm
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt


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


def slidingwindowsegment(sequence, create_segment, compute_error, max_error, seq_range=None, min_range=1):
    """
    Return a list of line segments that approximate the sequence.
    The list is computed using the sliding window technique. 
    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error
    """    
    if not seq_range:
        seq_range = (0,len(sequence))
    start = seq_range[0]      
    end = start                 

    result_segment = create_segment(sequence,(seq_range[0],seq_range[1]))
    
    while end < seq_range[1]: 
        end += 5              
        test_segment = create_segment(sequence,(start,end))
        error = compute_error(sequence,test_segment)
        if error <= max_error:
            result_segment = test_segment
        else:
            segment_range = result_segment[2] - result_segment[0] 
            while segment_range < min_range:

                end += 1
                test_segment = create_segment(sequence,(start,end))
                segment_range = test_segment[2] - test_segment[0]
            result_segment = test_segment
            break
    if end >= seq_range[1]: 
        return [result_segment]
    else:
        return [result_segment] + slidingwindowsegment(sequence, create_segment, compute_error, max_error, seq_range = (end-1,seq_range[1]))

    
def poly_regression(sequence, seq_range):
    """Return (x0,y0,x1,y1) of a poly_line fit to a segment of a sequence using polynomial regression"""
    y = sequence.values[seq_range[0]:seq_range[1]]
    x = np.arange(seq_range[0], seq_range[1])
    if len(x) != len(y):
        x = np.arange(seq_range[0], seq_range[1])[:len(y)]
    fit = np.polyfit(x, y, 2)
    p2 = np.poly1d(fit)
    y0 = p2(seq_range[0])
    y1 = p2(seq_range[1])
    return (seq_range[0],y0,seq_range[1],y1)

def poly_sumsquared_error(sequence, segment):
    """Return the sum of squared errors for a least squares line fit of one segment of a sequence"""
    y = sequence.values[segment[0]:segment[2]]
    x = np.arange(segment[0], segment[2])
    if len(x) != len(y):
        x = np.arange(segment[0], segment[2])[:len(y)]
    fit, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, full = True)
    return residuals

def update_segments(data, segments, min_range):
    segments_l = []   
    for i in range(len(segments)):
        speed_changed = False
        enough_range = False
        y = data.values[segments[i][0]:segments[i][2]]
        x = np.arange(segments[i][0], segments[i][2])
        if len(x) != len(y):
            x = np.arange(segments[i][0], segments[i][2])[:len(y)]
        fit = np.polyfit(x, y, 2)
        p2 = np.poly1d(fit)
        p_dot = np.polyder(p2)
        if speed_varies(p_dot(x)):
            speed_changed = True
            _ , idx = find_nearest(p_dot(x), 0)
        if speed_changed and (x[idx] - segments[i][0] > min_range) and (segments[i][2] - x[idx]+1 > min_range):
            segments_l.append([segments[i][0], segments[i][1], x[idx], segments[i][3]])
            segments_l.append([x[idx]+1, segments[i][1], segments[i][2], segments[i][3]])
            enough_range = True            
        if not enough_range or not speed_changed:
            segments_l.append([segments[i][0], segments[i][1], segments[i][2], segments[i][3]])
        
        seg_t_l = [i[2] - i[0] for i in segments_l]
        for i in range(len(seg_t_l)):
            if seg_t_l[i] < min_range:
                seg_t_temp = [segments_l[i-1][0],0,segments_l[i][2],0]
                del segments_l[i-1:i+1]
                segments_l.insert(i-1, seg_t_temp)
    return segments_l

def speed_varies(x):
    if min(x) * max(x) < 0:
        return True
    else:
        return False
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def find_nearest_from_end(array, value):
    arr = array - value
    idx = -1
    i = arr[idx]
    while i > 0:
        idx -= 1
        i = arr[idx]
    return array[idx], len(array)+idx

def draw_segments(data, segments, ax, spl):
    ax.plot(np.arange(len(data))/30, data, 'ro', ms=5, alpha=.1)
    ax.plot(np.arange(len(data))/30, spl(np.arange(len(data))/30), 'k', lw=2)
    ax.set_xticks(np.arange(0, len(data.values)/30, 5))
    xcoords = spl.get_knots()
    ax.set_ylim(0, 120)
    for xc in xcoords:
        ax.axvline(x=xc, ls = '--', alpha=.2)
        
def clasify_segments(data, segments, spl):
    l = []
    vel_avg_l = []
    acc_avg_l = []
    cnt = 1
    x = np.arange(len(data))/30
    segments = [i[0]/30 for i in segments]
    xrange_l = []
    for i in range(len(segments)):
        if i+1 == len(segments):
            xrange = (segments[i], len(data)/30)
        elif i == 0 :
            xrange = (0, segments[i+1])
        else:
            xrange = (segments[i], segments[i+1])
        xrange_l.append(xrange)
    M_best = pickle.load(open("../Data/finalized_model.sav", 'rb'))
    for segment in xrange_l:
        x = np.arange(segment[0], segment[1])
        y = spl(x)
        if len(x) != len(y):
            x = np.arange(segment[0], segment[1])[:len(y)]
        dis_dif = y[-1] - y[0]
        vel = spl.derivative(n=1)(x)
        vel_dif = vel[-1] - vel[0]
        vel_avg = dis_dif / len(x)
        acc_avg = vel_dif / len(x)
        acc = np.mean(spl.derivative(n=2)(x))
        _, std = norm.fit(y)
        phase = gmm_label_regime(gmm_clustering_predict(M_best, np.array(vel_avg).reshape(-1, 1)))
        l.append((phase[0], cnt))
        vel_avg_l.append(vel_avg)
        acc_avg_l.append(acc_avg)
        cnt += 1
    classes = [i[0] for i in l]
    if not "Backing off" in classes and not "Following" in classes:
        l = ["flying pass"]*len(l)
    return vel_avg_l, acc_avg_l, l

def gmm_label_regime(array):
    l = []
    for i in array:
        if i == 1:
            l.append('Following')
            continue
        if i == 2:
            l.append('Backing off')
            continue
        if i == 0:
            l.append('Approaching')
            continue
    return np.array(l)

def gmm_clustering_predict(model, X):
    """
    X is a (N, 1) array
    """
    X = np.clip(X, -2.5, 2.5) 
    return model.predict(X)

def merge_seg(l, segments):
    if len(set(l)) == 1:
        segments = [[segments[0][0], 0, segments[-1][2], 0]]
        return [l[0]], segments
    cnt = 0
    n_l = []
    l = [i[0] for i in l]
    for idx, cls in enumerate(l):
        if idx == len(l)-1:
            n_l.append([segments[idx][0], 0, segments[idx][2], 0, cls])
            break
        cnt = idx
        while l[cnt+1] == l[idx]:
            cnt += 1
            if cnt+1 == len(l):
                break
        n_l.append([segments[idx][0], 0, segments[cnt][2], 0, cls])
    l = [i[2] for i in n_l]
    d_l = []
    for i in range(len(l)-1):
        if l[i] == l[i+1]:
            d_l.append(i+1)
    n_l = [v for i,v in enumerate(n_l) if i not in frozenset(d_l)]     
    segments = [i[:-1] for i in n_l]
    classes = [i[-1] for i in n_l]
    return classes, segments

def draw_vel(data, segments, ax, spl):
    ax2 = ax.twinx()
    x = np.arange(len(data))/30
    y = data
    ax2.plot(x, spl.derivative()(x), 'b', lw=2)
    ax2.set_ylim(-20, 20)
    ax2.legend(["vel"], loc='upper right', fancybox=True)
    ax2.grid(b=True, which='major', linestyle='--', axis='y', alpha = .5)
    ax2.tick_params(axis='y', labelcolor='b')
                  
def draw_acc(data, segments, ax, spl):
    ax2 = ax.twinx()
    x = np.arange(len(data))/30
    y = data
    ax2.plot(x, spl.derivative(n=2)(x), 'g', lw=2)
    ax2.grid(b=True, which='major', linestyle='--', axis='y', alpha = .5)
    ax2.set_ylim(-20, 20)
    ax2.legend(["acc"], loc='upper center', bbox_to_anchor=(0.6, -0.05),
          fancybox=True, shadow=True, ncol=5)
    ax2.tick_params(axis='y', labelcolor='g')
    
def detect_phases(data, plot):
    avg_dist = smooth_scatter(data, 30)
    segments = slidingwindowsegment(avg_dist, poly_regression, poly_sumsquared_error, max_error=10, min_range=30)   
    segments = update_segments(avg_dist, segments, min_range=30)
    x = np.arange(len(data))/30
    y = data
    t = [i[0]/30 for i in segments]
    spl = LSQUnivariateSpline(x, y, t[1:], k=3)
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    if plot:
        draw_segments(avg_dist, segments, ax, spl)
        draw_vel(avg_dist, segments, ax, spl)
    vel, acc, classes = clasify_segments(avg_dist, segments, spl)
    classes, segments = merge_seg(classes, segments)
    vel, acc, classes = clasify_segments(avg_dist, segments, spl)
    classes, segments = merge_seg(classes, segments)
    if plot:
        for i in range(len(segments)):
            ax.text((segments[i][0]/30 + segments[i][2]/30)/2, 20, classes[i][0][0] + ", {}".format(i+1), rotation=90, alpha=1)
            ax.axvline(x = segments[i][0]/30, ls = '--', alpha=.4, c='k')
        ax.axvline(x = segments[-1][2]/30, ls = '--', alpha=.8, c='k')
    
    p = pd.Series([None]*len(data))
    for name,segment  in zip(classes, segments):
        if name == 'Approaching':
            p.iloc[segment[0]:segment[2]] = 'Approaching'
        elif name == 'Backing off':
            p.iloc[segment[0]:segment[2]] = 'Backing off'
        elif name == 'flying pass':
            p.iloc[segment[0]:segment[2]] = 'flying pass'
        else:
            p.iloc[segment[0]:segment[2]] = 'Following'
    return p, classes, segments
