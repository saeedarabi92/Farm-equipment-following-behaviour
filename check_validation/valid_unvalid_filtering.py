from utils.helper import generate_valid_unvalid_data, prepare_df, check_valid, dist_cal, generate_vid_csv_data, get_frame
import glob
import pandas as pd
import pathlib
import cv2

file_wd = str(pathlib.Path(__file__).parent.absolute())
video_folder = file_wd + '/videos/2018_videos'
csv_folder = file_wd + '/CSVs/2018_CSVs'
video_files = glob.glob(video_folder + '/*')
csv_files = glob.glob(csv_folder + '/*')
valid_frame_folder = file_wd + \
    '/valid_unvalid_files/2018_valid_unvalid_frames/valid_frames'
unvalid_frame_folder = file_wd + \
    '/valid_unvalid_files/2018_valid_unvalid_frames/unvalid_frames'
video_folder = file_wd + '/videos/2018_videos'

data = generate_vid_csv_data(csv_files, video_files)
v_l, uv_l = generate_valid_unvalid_data(data)

csv_file_names = [x.split('/')[-1].split('.')[0] for x in csv_files]
uv_l = [x.split('/')[-1].split('.')[0] for y, x in uv_l]
v_l = [x.split('/')[-1].split('.')[0] for x in v_l]


# print(v_l)
# print(uv_l)

for vid in video_files:
    action = ''
    vid_name = vid.split('/')[-1].split('.')[0]
    vid_csv = [x for x in csv_files if x.split(
        '/')[-1].split('.')[0] == vid_name]
    if len(vid_csv) != 0:
        df = pd.read_csv(vid_csv[0])
        if len(df) != 0:
            df['dist'] = df.H.apply(dist_cal)
            frame_num = df[df['dist'] == df['dist'].min()]['frame_num']
            fn = frame_num.values[0]
            frame = get_frame(vid, fn)
            if vid_name in v_l:
                cv2.imwrite(valid_frame_folder + '/' +
                            vid_name + '.jpeg', frame)
                action = 'saved into valid folder'
            elif vid_name in uv_l:
                cv2.imwrite(unvalid_frame_folder + '/' +
                            vid_name + '.jpeg', frame)
                action = 'saved into unvalid folder'
        else:
            action = ' has zero detection!'
    else:
        action = 'No csv found for this video!'
    print(vid, ' ', action)
