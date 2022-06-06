from operator import and_
import numpy as np
from PIL import Image
import pandas as pd
import os
import math
import csv
import pandas as pd
import mediapipe as mp
from PIL import Image
import numpy as np
import cv2 as cv2
import math
from utils_feature_extraction import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



results_hand_locations_list = ["hand_found_R", "hand_found_L", "origin_x_hand_L", "origin_y_hand_L", "origin_x_hand_R", "origin_y_hand_R",
          "thumb_finger_x_hand_L", "thumb_finger_y_hand_L", "thumb_finger_x_hand_R", "thumb_finger_y_hand_R",
  "index_finger_x_hand_L", "index_finger_y_hand_L", "index_finger_x_hand_R", "index_finger_y_hand_R",
 "middle_finger_x_hand_L", "middle_finger_y_hand_L", "middle_finger_x_hand_R", "middle_finger_y_hand_R",
 "ring_finger_x_hand_L", "ring_finger_y_hand_L", "ring_finger_x_hand_R", "ring_finger_y_hand_R", 
 "pinky_finger_x_hand_L", "pinky_finger_y_hand_L", "pinky_finger_x_hand_R", "pinky_finger_y_hand_R"]

x_columns = ["x_0","x_1","x_2","x_3","x_4","x_5","x_6","x_7","x_8","x_9","x_10","x_11","x_12","x_13","x_14","x_15","x_16","x_17","x_18","x_19","x_20","x_21","x_22","x_23","x_24","x_25","x_26","x_27","x_28","x_29","x_30","x_31","x_32","x_33","x_34","x_35","x_36","x_37","x_38","x_39","x_40","x_41","x_42","x_43","x_44","x_45","x_46","x_47","x_48","x_49","x_50","x_51","x_52","x_53","x_54","x_55","x_56","x_57","x_58","x_59","x_60","x_61","x_62","x_63","x_64","x_65","x_66","x_67"]
y_columns = ["y_0","y_1","y_2","y_3","y_4","y_5","y_6","y_7","y_8","y_9","y_10","y_11","y_12","y_13","y_14","y_15","y_16","y_17","y_18","y_19","y_20","y_21","y_22","y_23","y_24","y_25","y_26","y_27","y_28","y_29","y_30","y_31","y_32","y_33","y_34","y_35","y_36","y_37","y_38","y_39","y_40","y_41","y_42","y_43","y_44","y_45","y_46","y_47","y_48","y_49","y_50","y_51","y_52","y_53","y_54","y_55","y_56","y_57","y_58","y_59","y_60","y_61","y_62","y_63","y_64","y_65","y_66","y_67"]

openface_df = pd.read_csv("/content/openpose-research-keras/processed/all-batches.csv")
openface_df_90 = pd.read_csv("/content/openpose-research-keras/processed_90/all-batches-90.csv")
openface_df_270 = pd.read_csv("/content/openpose-research-keras/processed_270/all-batches-270.csv")
openface_df = addMinMax(openface_df)
openface_df_90 = addMinMax(openface_df_90)
openface_df_270 = addMinMax(openface_df_270)

#orientation_data = pd.read_csv('/content/openpose-research-keras/orientation.csv', names=["frame", "path", "orientation"], header=0)

pose_estimates_path = '/content/openpose-research-keras/data/estimates/'
#df = pd.read_pickle(os.path.join(pose_estimates_path, 'processed_pose_estimates_coords.pkl'))
#df = df.groupby(['video', "bp"]).apply(removeOutliers)

#df_real_pos = pd.read_pickle(os.path.join(pose_estimates_path, 'pose_estimates.pkl'))
#df_real_pos = df_real_pos.groupby(['video', "bp"]).apply(removeOutliers)

df_angle = pd.read_pickle(os.path.join(pose_estimates_path, 'processed_pose_estimates_angles.pkl'))

dir_path = '/content/drive/MyDrive/ResearchProject/data/all-batches/'
list_frame_images = sorted(os.listdir(dir_path))

target_path = '/content/openpose-research-keras/data_full_csv_v2.csv'
data_target = []
with open(target_path, newline='') as f:
    reader = csv.reader(f)
    data_target = list(reader)

data = []
c = 0


frame_num_old = 0
face_coords_final_old = None
frame_video_num = 0
first_frame = True
values_old_nose = None
values_old_neck = None
values_old_leye = None
values_old_reye = None
values_old_hip = None
values_old_lwrist = None
values_old_rwrist = None
values_old_dict = {
    "REye": None,
    "LEye": None,
    "RWrist": None,
    "LWrist": None,
    "RWrist_r": None,
    "LWrist_r": None,
    "REar": None,
    "LEar": None,
    "Neck": None,
    "Nose": None,
}
old_video_name = None

old_hand_values = {}
for n in results_hand_locations_list:
  old_hand_values[n] = np.nan

coord_hand_x_L_old = np.nan
coord_hand_x_R_old = np.nan
coord_hand_y_L_old = np.nan
coord_hand_y_R_old = np.nan

for image in list_frame_images:
  print(c)
  #if (c < 5745):
  #  c+=1
  #  continue;
  c+=1
  image_name = image
  split_img = image.split('_')
  image_num_ext = split_img[3] #index of extracted image: two images every second
  frame_num_orig = split_img[2] #index of frame in the whole video
  frame_num = int(frame_num_orig) - 1
  image_num = image_num_ext.split('.')[0]
  video_name = split_img[0] + "_" + split_img[1]
  #video_name_max = (video_name[:12]) if len(video_name) > 12 else video_name
  video_name_max = video_name
  print(video_name_max)
  print(image)
  print("===========")
  
  if (old_video_name != video_name or (frame_num - frame_num_old > 500)):
    print("change video------->>")
    frame_num_old = frame_num
    coordX_old = None
    coordY_old = None
    size_old = None
    face_coords_final_old = None
    frame_video_num = 0
    first_frame = True
    values_old_nose = None
    values_old_neck = None
    values_old_leye = None
    values_old_reye = None
    values_old_hip = None
    values_old_lwrist = None
    values_old_rwrist = None
    values_old_dict = {
        "REye": None,
        "LEye": None,
        "RWrist": None,
        "LWrist": None,
        "RWrist_r": None,
        "LWrist_r": None,
        "REar": None,
        "LEar": None,
        "Neck": None,
        "Nose": None,
    }
    coord_hand_x_L_old = np.nan
    coord_hand_x_R_old = np.nan
    coord_hand_y_L_old = np.nan
    coord_hand_y_R_old = np.nan
    for n in results_hand_locations_list:
      old_hand_values[n] = np.nan

  #Set values for Elbow angles
  values_LElbow = df_angle.loc[(df_angle['frame'] == int(frame_num)) & (df_angle['bp'] == 'LElbow') & (df_angle['video'] == video_name_max)]
  if not values_LElbow.empty:
    angle_LElbow = values_LElbow['angle'].iloc[0]
  else:
    angle_LElbow = None
  values_RElbow = df_angle.loc[(df_angle['frame'] == int(frame_num)) & (df_angle['bp'] == 'RElbow') & (df_angle['video'] == video_name_max)]
  if not values_RElbow.empty:
    angle_RElbow = values_RElbow['angle'].iloc[0]
  else:
    angle_RElbow = None

  #Check frame number exists on data, else use previous frame number
  df_frame = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['video'] == video_name_max)]
  while (df_frame.empty):
    frame_num = frame_num - 1
    df_frame = df_real_pos.loc[(df['frame'] == int(frame_num)) & (df['video'] == video_name_max)]

  #Set real values for Eyes and Nose
  values_LEye_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'LEye') & (df_real_pos['video'] == video_name_max)]
  values_REye_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'REye') & (df_real_pos['video'] == video_name_max)]
  values_Nose_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'Nose') & (df_real_pos['video'] == video_name_max)]

  coord_LEyeX_r, coord_LEyeY_r = (values_LEye_r['x'].iloc[0], values_LEye_r['y'].iloc[0])
  coord_REyeX_r, coord_REyeY_r = (values_REye_r['x'].iloc[0], values_REye_r['y'].iloc[0])
  coord_NoseX_r, coord_NoseY_r = (values_Nose_r['x'].iloc[0], values_Nose_r['y'].iloc[0])

  #Set real values for Wrists
  values_LWrist_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'LWrist') & (df_real_pos['video'] == video_name_max)]
  values_RWrist_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'RWrist') & (df_real_pos['video'] == video_name_max)]

  coord_LWristX_r, coord_LWristY_r = (values_LWrist_r['x'].iloc[0], values_LWrist_r['y'].iloc[0])
  coord_RWristX_r, coord_RWristY_r = (values_RWrist_r['x'].iloc[0], values_RWrist_r['y'].iloc[0])
  
  #Set values for Neck and Hips
  values_Neck = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'Neck') & (df_real_pos['video'] == video_name_max)]
  values_LHip = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'LHip') & (df_real_pos['video'] == video_name_max)]
  values_RHip = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == 'RHip') & (df_real_pos['video'] == video_name_max)]
  coord_NeckX, coord_NeckY = (values_Neck['x'].iloc[0], values_Neck['y'].iloc[0])
  coord_LHipX, coord_LHipY = (values_LHip['x'].iloc[0], values_LHip['y'].iloc[0])
  coord_RHipX, coord_RHipY = (values_RHip['x'].iloc[0], values_RHip['y'].iloc[0])
  
  #If Neck value is none, use previous value or search in next frames
  if (np.isnan(coord_NeckX)):
    if (values_old_neck is not None):
      values_Neck = values_old_neck
      coord_NeckX, coord_NeckY = (values_Neck['x'].iloc[0], values_Neck['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coord_NeckX):
          frame_num_now +=1
          values_Neck = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'Neck') & (df_real_pos['video'] == video_name_max)]
          coord_NeckX, coord_NeckY = (values_Neck['x'].iloc[0], values_Neck['y'].iloc[0])

  #If Nose value is none, use previous value or search in next frames
  if (np.isnan(coord_NoseX_r)) :
    if (values_old_nose is not None):
      values_Nose_r = values_old_nose
      coord_NoseX_r, coord_NoseY_r = (values_Nose_r['x'].iloc[0], values_Nose_r['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coord_NoseX_r):
          frame_num_now +=1
          values_Nose_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'Nose') & (df_real_pos['video'] == video_name_max)]
          if values_Nose_r.empty:
            coord_NoseX_r, coord_NoseY_r = (np.NaN, np.NaN)
          else:
            coord_NoseX_r, coord_NoseY_r = (values_Nose_r['x'].iloc[0], values_Nose_r['y'].iloc[0])
  
  #If Eye values are none, use previous value or search in next frames  
  if (np.isnan(coord_LEyeX_r)):
    if (values_old_leye is not None):
      values_LEye_r = values_old_leye
      coord_LEyeX_r, coord_LEyeY_r = (values_LEye_r['x'].iloc[0], values_LEye_r['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coord_LEyeX_r):
          frame_num_now +=1
          values_LEye_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'LEye') & (df_real_pos['video'] == video_name_max)]
          if values_LEye_r.empty:
            coord_LEyeX_r, coord_LEyeY_r = (np.NaN, np.NaN)
          else:
            coord_LEyeX_r, coord_LEyeY_r = (values_LEye_r['x'].iloc[0], values_LEye_r['y'].iloc[0])

  if (np.isnan(coord_REyeX_r)):
    if (values_old_reye is not None):
      values_REye_r = values_old_reye
      coord_REyeX_r, coord_REyeY_r = (values_REye_r['x'].iloc[0], values_REye_r['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coord_REyeX_r):
          frame_num_now +=1
          values_REye_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'REye') & (df_real_pos['video'] == video_name_max)]
          if values_REye_r.empty:
            coord_REyeX_r, coord_REyeY_r = (np.NaN, np.NaN)
          else:
            coord_REyeX_r, coord_REyeY_r = (values_REye_r['x'].iloc[0], values_REye_r['y'].iloc[0])

  #If Wrist values are none, use previous value or search in next frames  
  if (np.isnan(coord_LWristX_r)):
    if ((values_old_lwrist is not None) and not (values_old_lwrist.empty)):
      values_LWrist_r = values_old_lwrist
      coord_LWristX_r, coord_LWristY_r = (values_LWrist_r['x'].iloc[0], values_LWrist_r['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coord_LWristX_r) and frame_num_now < 8000:
          frame_num_now +=1
          values_LWrist_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'LWrist') & (df_real_pos['video'] == video_name_max)]
          if values_LWrist_r.empty:
            coord_LWristX_r, coord_LWristY_r = (np.NaN, np.NaN)
          else:
            coord_LWristX_r, coord_LWristY_r = (values_LWrist_r['x'].iloc[0], values_LWrist_r['y'].iloc[0])

  if (np.isnan(coord_RWristX_r)):
    if ((values_old_rwrist is not None) and not (values_old_rwrist.empty)):
      values_RWrist_r = values_old_rwrist
      coord_RWristX_r, coord_RWristY_r = (values_RWrist_r['x'].iloc[0], values_RWrist_r['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coord_RWristX_r) and frame_num_now < 8000:
          frame_num_now +=1
          values_RWrist_r = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'RWrist') & (df_real_pos['video'] == video_name_max)]
          if values_RWrist_r.empty:
            coord_RWristX_r, coord_RWristY_r = (np.NaN, np.NaN)
          else:
            coord_RWristX_r, coord_RWristY_r = (values_RWrist_r['x'].iloc[0], values_RWrist_r['y'].iloc[0])

  #If both Hip values are none, use previous values or search in next frames  
  values_Hip = values_LHip
  coord_HipX, coord_HipY = (values_Hip['x'].iloc[0], values_Hip['y'].iloc[0])
  if(np.isnan(coord_LHipX)):
    if (np.isnan(coord_RHipX)):
      if (values_old_hip is not None):
        values_Hip = values_old_hip
      else:
        frame_num_now = int(frame_num)
        values_Hip = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'LHip') & (df_real_pos['video'] == video_name_max)]
        while np.isnan(coord_HipX) and not values_LHip.empty:
          frame_num_now +=1
          values_LHip = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'LHip') & (df_real_pos['video'] == video_name_max)]
          values_Hip = values_LHip
          if (values_Hip.empty):
            values_Hip = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'RHip') & (df_real_pos['video'] == video_name_max)]
          if not values_Hip.empty:
            coord_HipX, coord_HipY = (values_Hip['x'].iloc[0], values_Hip['y'].iloc[0])
    else:
      values_Hip = values_RHip
  else:
    values_Hip = values_LHip
  
  if not values_Hip.empty:
    coord_HipX, coord_HipY = (values_Hip['x'].iloc[0], values_Hip['y'].iloc[0])
  else:
    coord_HipX, coord_HipY = (None, None)
  
  #Get values of width and height of the frame
  values_left_eye_pixel_x_y = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LEye') & (df['video'] == video_name_max)]
  pixel_x, pixel_y = (values_left_eye_pixel_x_y['pixel_x'].iloc[0], values_left_eye_pixel_x_y['pixel_y'].iloc[0])

  #Get Box coordinates around Hip and Neck
  path = '/content/drive/MyDrive/ResearchProject/data/all-batches/' + image
  orientation = orientation_data.loc[(orientation_data['path'] == path)]["orientation"].iloc[0]
  coordX, coordY, size = getBoxCoordinates(coord_HipX, coord_HipY, coord_NeckX, coord_NeckY, pixel_x, pixel_y, orientation)
  if coordX is None:
    coordX = coordX_old
    coordY = coordY_old
    size = size_old
  if coordX is None:
    coordX = 0
    coordY = 0
    size = pixel_x
  
  #Get face coordinates from OpenFace
  if orientation == 90:
    openface_df_faces_90 = openface_df_90.loc[openface_df_90['frame'] == c]
  elif orientation == 270:
    openface_df_faces_270 = openface_df.loc[openface_df['frame'] == c]
  else:
    openface_df_faces = openface_df_270.loc[openface_df_270['frame'] == c]

  #Check which faces are inside the box using OpenPose and OpenFace
  y_max = int(pixel_x)
  x_max = int(pixel_y)
  faces = []
  faces_90 = []
  faces_270 = []
  if orientation == 90:
    faces_90 = getFaces(openface_df_faces_90, y_max, x_max, (coord_LEyeX_r, coord_LEyeY_r), (coord_REyeX_r, coord_REyeY_r), (coord_NeckX, coord_NeckY), coordX, coordY, size, 90)
  elif orientation == 270:
    faces_270 = getFaces(openface_df_faces_270, y_max, x_max, (coord_LEyeX_r, coord_LEyeY_r), (coord_REyeX_r, coord_REyeY_r), (coord_NeckX, coord_NeckY), coordX, coordY, size, 270)
  else:
    faces = getFaces(openface_df_faces, y_max, x_max, (coord_LEyeX_r, coord_LEyeY_r), (coord_REyeX_r, coord_REyeY_r), (coord_NeckX, coord_NeckY), coordX, coordY, size, 0)

  face_coords_final = None
  if (values_old_leye is not None) and (values_old_reye is not None):
    oldEyes = True
  else:
    oldEyes = False
  if (len(faces) > 0 or len(faces_90) > 0 or len(faces_270) > 0):
    face_coords_final = getFaceCoordsFinal(faces, faces_90, faces_270, coord_NeckX, coord_NeckY, orientation, oldEyes)

  if face_coords_final is None:
    face_coords_final = face_coords_final_old
    c_now = c
    not_changed_video = True
    check_frame = 0
    while face_coords_final is None and not_changed_video:
      c_now += 1
      split_img_now = list_frame_images[c_now].split('_')
      video_name_now = split_img_now[0] + "_" + split_img_now[1]
      if orientation == 90:
        openface_df_faces_now_90 = openface_df_90.loc[openface_df_90['frame'] == c_now]
      elif orientation == 270:
        openface_df_faces_now_270 = openface_df_270.loc[openface_df_270['frame'] == c_now]
      else:
        openface_df_faces_now = openface_df.loc[openface_df['frame'] == c_now]
      y_max = int(pixel_x)
      x_max = int(pixel_y)
      if orientation == 90:
        faces_90, not_changed_video = getFaces(openface_df_faces_now_90, y_max, x_max, (coord_LEyeX_r, coord_LEyeY_r), (coord_REyeX_r, coord_REyeY_r), (coord_NeckX, coord_NeckY), coordX, coordY, size, 0, True, video_name_max, video_name_now)
      elif orientation == 270:
        faces_270, not_changed_video = getFaces(openface_df_faces_now_270, y_max, x_max, (coord_LEyeX_r, coord_LEyeY_r), (coord_REyeX_r, coord_REyeY_r), (coord_NeckX, coord_NeckY), coordX, coordY, size, 0, True, video_name_max, video_name_now)
      else:
        faces, not_changed_video = getFaces(openface_df_faces_now, y_max, x_max, (coord_LEyeX_r, coord_LEyeY_r), (coord_REyeX_r, coord_REyeY_r), (coord_NeckX, coord_NeckY), coordX, coordY, size, 0, True, video_name_max, video_name_now)
      if (len(faces) > 0 or len(faces_90) > 0 or len(faces_270) > 0):
        face_coords_final = getFaceCoordsFinal(faces, faces_90, faces_270, coord_NeckX, coord_NeckY, orientation, oldEyes)

  #Ref Distance
  if (coord_HipX is not None and coord_NeckX is not None):
    ref_dist = math.sqrt((coord_HipX - coord_NeckX)**2 + (coord_HipY - coord_NeckY)**2)
  else:
    ref_dist = np.nan

  #Set bounding box based on face coordinates
  if face_coords_final is None:
    print("is none")
  else:
    coordX_min = face_coords_final[0]
    coordY_min = face_coords_final[1]
    coordX_max = face_coords_final[2]
    coordY_max = face_coords_final[3]
    sizeX = coordX_max - coordX_min
    sizeY = coordY_max - coordY_min
    size = max(sizeX, sizeY)
    coordX = coordX_min - size*0.5
    coordY = coordY_min - size*0.5
    size*=2
    if coordX + size > int(pixel_x):
        coordX = int(pixel_x) - size
    if coordY + size > int(pixel_y):
        coordY = int(pixel_y) - size
    if coordY < 0:
        coordY = 0
    if (coordX < 0):
        coordX = 0
    if size > int(pixel_y):
        size = int(pixel_y)

  values_list = ["LEar", "REar", "LEye", "REye", "LWrist", "RWrist", "Nose", "Neck"]
  values_dict = {}
  for i in values_list:
    values_dict[i] = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == i) & (df['video'] == video_name_max)]

  #values_dict["LWrist_r"] =  df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == "LWrist") & (df_real_pos['video'] == video_name_max)]
  #values_dict["RWrist_r"] =  df_real_pos.loc[(df_real_pos['frame'] == int(frame_num)) & (df_real_pos['bp'] == "RWrist") & (df_real_pos['video'] == video_name_max)]

  coords = {}
  coords["coord_LEarX"], coords["coord_LEarY"] = (values_dict["LEar"]['x'].iloc[0], values_dict["LEar"]['y'].iloc[0])
  coords["coord_REarX"], coords["coord_REarY"] = (values_dict["REar"]['x'].iloc[0], values_dict["REar"]['y'].iloc[0])

  coords["coord_REyeX"], coords["coord_REyeY"]= (values_dict["REye"]['x'].iloc[0], values_dict["REye"]['y'].iloc[0])
  coords["coord_LEyeX"], coords["coord_LEyeY"]= (values_dict["LEye"]['x'].iloc[0], values_dict["LEye"]['y'].iloc[0])
  if(face_coords_final is not None):
    print("=============")
    print("USING OPENFACE COORDS")
    print("=============")
    coords["coord_REyeX_r"], coords["coord_REyeY_r"] = (face_coords_final[4], face_coords_final[5])
    coords["coord_LEyeX_r"], coords["coord_LEyeY_r"] = (face_coords_final[6], face_coords_final[7])

  #validate openface eyes fit with skeleton coordinates
  if (np.isnan(coord_REyeX_r) or coordX_old is None):
    print("ok_1_1")
  elif (coord_REyeX_r - 20 > coordX and (coord_REyeX_r + 20 < coordX + size)):
    print("ok_1_2")
  elif (coord_REyeX_r - 20 > coordX_old and (coord_REyeX_r + 20 < coordX_old + size_old)):
    print ("exchanging coords_1")
    coords["coord_REyeX_r"], coords["coord_REyeY_r"] = (coord_REyeX_r, coord_REyeY_r)
    coords["coord_LEyeX_r"], coords["coord_LEyeY_r"] = (coord_LEyeX_r, coord_LEyeY_r)
    coordX = coordX_old
    coordY = coordY_old
    #values_LEye_r = (coord_LEyeX_r, coord_LEyeY_r)
    #valuesREye_r = (coord_REyeX_r, coord_REyeY_r)
    size = size_old
  
  if (np.isnan(coord_REyeY_r) or coordY_old is None):
    print("ok_1_1")
  elif (coord_REyeY_r - 20 > coordY and (coord_REyeY_r + 20 < coordY + size)):
    print("ok_1_2")
  elif (coord_REyeY_r - 20 > coordY_old and (coord_REyeY_r + 20 < coordY_old + size_old)):
    print ("exchanging coords_1")
    coords["coord_REyeX_r"], coords["coord_REyeY_r"] = (coord_REyeX_r, coord_REyeY_r)
    coords["coord_LEyeX_r"], coords["coord_LEyeY_r"] = (coord_LEyeX_r, coord_LEyeY_r)
    coordX = coordX_old
    coordY = coordY_old
    #values_LEye_r = (coord_LEyeX_r, coord_LEyeY_r)
    #valuesREye_r = (coord_REyeX_r, coord_REyeY_r)
    size = size_old

  if (np.isnan(coord_LEyeX_r) or coordX_old is None):
    print("ok_2_1")
  elif (coord_LEyeX_r - 20 > coordX and (coord_LEyeX_r + 20 < coordX + size)):
    print("ok_2_2")
  elif (coord_LEyeX_r - 20 > coordX_old and (coord_LEyeX_r + 20 < coordX_old + size_old)):
    print ("exchanging coords_2")
    coords["coord_REyeX_r"], coords["coord_REyeY_r"] = (coord_REyeX_r, coord_REyeY_r)
    coords["coord_LEyeX_r"], coords["coord_LEyeY_r"] = (coord_LEyeX_r, coord_LEyeY_r)
    coordX = coordX_old
    coordY = coordY_old
    #values_LEye_r = (coord_LEyeX_r, coord_LEyeY_r)
    #valuesREye_r = (coord_REyeX_r, coord_REyeY_r)
    size = size_old
  
  if (np.isnan(coord_LEyeY_r) or coordY_old is None):
    print("ok_2_1")
  elif (coord_LEyeY_r - 20 > coordY and (coord_LEyeY_r + 20 < coordY + size)):
    print("ok_2_2")
  elif (coord_LEyeY_r - 20 > coordY_old and (coord_LEyeY_r + 20 < coordY_old + size_old)):
    print ("exchanging coords_2")
    coords["coord_REyeX_r"], coords["coord_REyeY_r"] = (coord_REyeX_r, coord_REyeY_r)
    coords["coord_LEyeX_r"], coords["coord_LEyeY_r"] = (coord_LEyeX_r, coord_LEyeY_r)
    coordX = coordX_old
    coordY = coordY_old
    #values_LEye_r = (coord_LEyeX_r, coord_LEyeY_r)
    #valuesREye_r = (coord_REyeX_r, coord_REyeY_r)
    size = size_old
  img = Image.open(path)
  box = coordX, coordY, coordX+size, coordY + size
  img_2 = img.crop(box)
  img_2 = img_2.resize((224,224))
    
  img = Image.open(path)
  img = img.convert('RGB')
  img = alignFace([coords["coord_REyeX_r"], coords["coord_REyeY_r"]], [coords["coord_LEyeX_r"], coords["coord_LEyeY_r"]], (coord_NeckX, coord_NeckY), size, np.array(img), orientation, x_max, y_max)
  path_cropped = "/content/drive/MyDrive/ResearchProject/cropped3/cropped_" + str(video_name) + "_"  + str(frame_num_orig) + "_" + str(image_num) + "_" + str(orientation) + ".jpg"
  Image.fromarray(img).save(path_cropped)
  img_2.save("/content/drive/MyDrive/ResearchProject/cropped3/cropped_" + str(video_name) + "_"  + str(frame_num_orig) + "_" + str(image_num) + "_" + str(orientation) + "_2.jpg")
  
  coords["coord_LWristX"], coords["coord_LWristY"] = (values_dict["LWrist"]['x'].iloc[0], values_dict["LWrist"]['y'].iloc[0])
  coords["coord_RWristX"], coords["coord_RWristY"] = (values_dict["RWrist"]['x'].iloc[0], values_dict["RWrist"]['y'].iloc[0])
  #coords["coord_LWristX_r"], coords["coord_LWristY_r"] = (values_dict["LWrist_r"]['x'].iloc[0], values_dict["LWrist_r"]['y'].iloc[0])
  #coords["coord_RWristX_r"], coords["coord_RWristY_r"] = (values_dict["RWrist_r"]['x'].iloc[0], values_dict["RWrist_r"]['y'].iloc[0])

  coords["coord_NoseX"], coords["coord_NoseY"] = (values_dict["Nose"]['x'].iloc[0], values_dict["Nose"]['y'].iloc[0])
  coords["coord_NeckX"], coords["coord_NeckY"] = (values_dict["Neck"]['x'].iloc[0], values_dict["Neck"]['y'].iloc[0])

  for i in values_list:
    if (np.isnan(coords["coord_" + i + "X"])):
      if (values_old_dict[i] is not None):
        values_dict[i] = values_old_dict[i]
        if not values_dict[i].empty:
          coords["coord_"+ i + "X"], coords["coord_" + i + "Y"] = (values_dict[i]['x'].iloc[0], values_dict[i]['y'].iloc[0])
      else:
        frame_num_now = int(frame_num)
        if (first_frame):
          while np.isnan(coords["coord_" + i + "X"]) and not values_dict[i].empty:
            frame_num_now +=1
            values_dict[i] = df.loc[(df['frame'] == int(frame_num_now)) & (df['bp'] == i) & (df['video'] == video_name_max)]
            if not values_dict[i].empty:
              coords["coord_" + i + "X"], coords["coord_" + i + "Y"] = (values_dict[i]['x'].iloc[0], values_dict[i]['y'].iloc[0])
    values_old_dict[i] = values_dict[i]
  
  '''
  if (np.isnan(coords["coord_LWristX_r"])):
    if (values_old_dict["LWrist_r"] is not None):
      values_dict["LWrist_r"] = values_old_dict["LWrist_r"]
      coords["coord_LWristX_r"], coords["coord_LWristY_r"] = (values_dict["LWrist_r"]['x'].iloc[0], values_dict["LWrist_r"]['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coords["coord_LWristX_r"]) and not values_dict["LWrist_r"].empty:
          frame_num_now +=1
          values_dict["LWrist_r"] = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'LWrist') & (df_real_pos['video'] == video_name_max)]
          if not values_dict["LWrist_r"].empty:
            coords["coord_LWristX_r"], coords["coord_LWristY_r"] = (values_dict["LWrist_r"]['x'].iloc[0], values_dict["LWrist_r"]['y'].iloc[0])
  values_old_dict["LWrist_r"] = values_dict["LWrist_r"]
  
  if (np.isnan(coords["coord_RWristX_r"])):
    if ((values_old_dict["RWrist_r"] is not None) and not values_old_dict["RWrist_r"].empty):
      values_dict["RWrist_r"] = values_old_dict["RWrist_r"]
      coords["coord_RWristX_r"], coords["coord_RWristY_r"] = (values_dict["RWrist_r"]['x'].iloc[0], values_dict["RWrist_r"]['y'].iloc[0])
    else:
      frame_num_now = int(frame_num)
      if (first_frame):
        while np.isnan(coords["coord_RWristX_r"]) and not values_dict["RWrist_r"].empty:
          frame_num_now +=1
          values_dict["RWrist_r"] = df_real_pos.loc[(df_real_pos['frame'] == int(frame_num_now)) & (df_real_pos['bp'] == 'RWrist') & (df_real_pos['video'] == video_name_max)]
          if not values_dict["RWrist_r"].empty:
            coords["coord_RWristX_r"], coords["coord_RWristY_r"] = (values_dict["RWrist_r"]['x'].iloc[0], values_dict["RWrist_r"]['y'].iloc[0])
  values_old_dict["RWrist_r"] = values_dict["RWrist_r"]
  '''
  
  first_frame = False

  speed_LWrist = None
  speed_RWrist = None
  if(not values_dict["LWrist"]['speed'].empty):
    speed_LWrist = values_dict["LWrist"]['speed'].iloc[0]
  if (not values_dict["RWrist"]['speed'].empty):
    speed_RWrist = values_dict["RWrist"]['speed'].iloc[0]

  L1_dist_x = coords["coord_LEarX"]-coords["coord_LWristX"]
  L1_dist_y = coords["coord_LEarY"]-coords["coord_LWristY"]
  L1_dist = math.sqrt(pow(L1_dist_x,2) + pow(L1_dist_y,2)) # Distance Left Wrist - Left Ear
  L2_dist_x = coords["coord_REarX"]-coords["coord_RWristX"]
  L2_dist_y = coords["coord_REarY"]-coords["coord_RWristY"] # Distance Right Wrist - Right Ear
  L2_dist = math.sqrt(pow(L2_dist_x,2) + pow(L2_dist_y,2)) # Distance Right Wrist - Right Ear
  if(face_coords_final is not None):
    L3_dist_x = coords["coord_LEyeX_r"]-coord_LWristX_r
    L3_dist_y = coords["coord_LEyeY_r"]-coord_LWristY_r
    L3_dist = math.sqrt(pow(L3_dist_x,2) + pow(L3_dist_y,2)) # Distance Left Wrist - Left Eye
    
    L4_dist_x = coords["coord_REyeX_r"]-coord_RWristX_r
    L4_dist_y = coords["coord_REyeY_r"]-coord_RWristY_r 
    L4_dist = math.sqrt(pow(L4_dist_x,2) + pow(L4_dist_y,2)) # Distance Right Wrist - Right Eye
    if (not values_dict["LEye"].empty):
      L3_dist = L3_dist
      L4_dist = L4_dist
    elif (not values_dict["LEar"].empty):
      L3_dist = L3_dist
      L4_dist = L4_dist
    else:
      L3_dist = None
      L4_dist = None
  else:
    L3_dist = None
    L4_dist = None
  
  L5_dist_x = coords["coord_NoseX"]-coords["coord_LWristX"]
  L5_dist_y = coords["coord_NoseY"]-coords["coord_LWristY"]
  L5_dist = math.sqrt(pow(L5_dist_x,2) + pow(L5_dist_y,2)) # Distance Left Wrist - Nose

  L6_dist_x = coords["coord_NoseX"]-coords["coord_RWristX"]
  L6_dist_y = coords["coord_NoseY"]-coords["coord_RWristY"]
  L6_dist = math.sqrt(pow(L6_dist_x,2) + pow(L6_dist_y,2)) # Distance Right Wrist - Nose

  L7_dist_x = coords["coord_NeckX"]-coords["coord_LWristX"]
  L7_dist_y = coords["coord_NeckY"]-coords["coord_LWristY"]
  L7_dist = math.sqrt(pow(L7_dist_x,2) + pow(L7_dist_y,2)) # Distance Left Wrist - Neck

  L8_dist_x = coords["coord_NeckX"]-coords["coord_RWristX"]
  L8_dist_y = coords["coord_NeckY"]-coords["coord_RWristY"]
  L8_dist = math.sqrt(pow(L8_dist_x,2) + pow(L8_dist_y,2)) # Distance Right Wrist - Neck

  L9_dist_x = coords["coord_LEyeX"]-coords["coord_LWristX"]
  L9_dist_y = coords["coord_LEyeY"]-coords["coord_LWristY"]
  L9_dist = math.sqrt(pow(L9_dist_x,2) + pow(L9_dist_y,2)) # Distance Left Wrist - Left Eye - L12

  L10_dist_x = coords["coord_REyeX"]-coords["coord_LWristX"]
  L10_dist_y = coords["coord_REyeY"]-coords["coord_LWristY"]
  L10_dist = math.sqrt(pow(L10_dist_x,2) + pow(L10_dist_y,2)) # Distance Left Wrist - Right Eye - L11

  L11_dist_x = coords["coord_LEyeX"]-coords["coord_RWristX"]
  L11_dist_y = coords["coord_LEyeY"]-coords["coord_RWristY"]
  L11_dist = math.sqrt(pow(L11_dist_x,2) + pow(L11_dist_y,2)) # Distance Right Wrist - Left Eye - L10
  
  L12_dist_x = coords["coord_REyeX"]-coords["coord_RWristX"]
  L12_dist_y = coords["coord_REyeY"]-coords["coord_RWristY"]
  L12_dist = math.sqrt(pow(L12_dist_x,2) + pow(L12_dist_y,2)) # Distance Right Wrist - Right Eye - L9


  displacement_RWrist = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'RWrist') & (df['video'] == video_name_max)]['displacement'].iloc[0]
  displacement_LWrist = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LWrist') & (df['video'] == video_name_max)]['displacement'].iloc[0]
  displacement_RElbow = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'RElbow') & (df['video'] == video_name_max)]['displacement'].iloc[0]
  displacement_LElbow = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LElbow') & (df['video'] == video_name_max)]['displacement'].iloc[0]
  speed_RElbow = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'RElbow') & (df['video'] == video_name_max)]['speed'].iloc[0]
  speed_LElbow = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LElbow') & (df['video'] == video_name_max)]['speed'].iloc[0]
  acceleration_RWrist_x = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'RWrist') & (df['video'] == video_name_max)]['acceleration_x'].iloc[0]
  acceleration_RWrist_y = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'RWrist') & (df['video'] == video_name_max)]['acceleration_y'].iloc[0]
  acceleration_LWrist_x = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LWrist') & (df['video'] == video_name_max)]['acceleration_x'].iloc[0]
  acceleration_LWrist_y = df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LWrist') & (df['video'] == video_name_max)]['acceleration_y'].iloc[0]
  coord_REye_Exists = np.isnan(df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'REye') & (df['video'] == video_name_max)]['x'].iloc[0]) == False
  coord_LEye_Exists = np.isnan(df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LEye') & (df['video'] == video_name_max)]['x'].iloc[0]) == False
  coord_REar_Exists = np.isnan(df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'REar') & (df['video'] == video_name_max)]['x'].iloc[0]) == False
  coord_LEar_Exists = np.isnan(df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'LEar') & (df['video'] == video_name_max)]['x'].iloc[0]) == False
  coord_Nose_Exists = np.isnan(df.loc[(df['frame'] == int(frame_num)) & (df['bp'] == 'Nose') & (df['video'] == video_name_max)]['x'].iloc[0]) == False

  row_data = [L1_dist, L1_dist_x, L1_dist_y, L2_dist, L2_dist_x, L2_dist_y, L3_dist, L3_dist_x, L3_dist_y, L4_dist, L4_dist_x, L4_dist_y, L5_dist, L5_dist_x, L5_dist_y, L6_dist, L6_dist_x, L6_dist_y, L7_dist, L7_dist_x, L7_dist_y, L8_dist, L8_dist_x, L8_dist_y, L9_dist, L9_dist_x, L9_dist_y, L10_dist, L10_dist_x, L10_dist_y, L11_dist, L11_dist_x, L11_dist_y, L12_dist, L12_dist_x, L12_dist_y, speed_LWrist, speed_RWrist, angle_LElbow, angle_RElbow, displacement_RWrist, displacement_LWrist, displacement_RElbow, displacement_LElbow, speed_RElbow, speed_LElbow, acceleration_RWrist_x, acceleration_RWrist_y, acceleration_LWrist_x, acceleration_LWrist_y, coord_REye_Exists, coord_LEye_Exists, coord_REar_Exists, coord_LEar_Exists, coord_Nose_Exists, coordX, coordY, size, orientation, path, path_cropped]

  #Hand and Finger detection
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.4) as hands:
      
    file_full = '/content/drive/MyDrive/ResearchProject/data/all-batches/' + image
    sizeX = sizeY = None
    if coord_LWristX_r and coord_RWristX_r and not np.isnan(coord_RWristX_r) and not np.isnan(coord_LWristX_r):
      coordX, coordX_max = min(coord_LWristX_r, coord_RWristX_r) - 100.0/1280*pixel_x, max(coord_LWristX_r, coord_RWristX_r) + 100.0/1280*pixel_x
      coordY, coordY_max = min(coord_LWristY_r, coord_RWristY_r) - 100.0/960*pixel_y, max(coord_LWristY_r, coord_RWristY_r) + 100.0/960*pixel_y
      sizeX = coordX_max - coordX
      sizeY = coordY_max - coordY

    results_hand_locations = getHandData(file_full, orientation, size, coordX, coordY, sizeX, sizeY, c)
  
  id_c = 0
  for i in results_hand_locations:
    if id_c == 0:
      hand_found_R = i
      row_data.append(hand_found_R)
    elif id_c == 1:
      hand_found_L = i
      row_data.append(hand_found_L)
    else:
      if (id_c%2 == 0): #x coordinate
        value_x = i
        value_y = results_hand_locations[id_c + 1]
        if (np.isnan(i)):
          old_value_x = old_hand_values[results_hand_locations_list[id_c]]
          old_value_y = old_hand_values[results_hand_locations_list[id_c+1]]
          if np.isnan(old_value_x):
            L1_dist_x = L1_dist_y = L1_dist = np.nan
            L2_dist_x = L2_dist_y = L2_dist = np.nan
            L3_dist_x = L3_dist_y = L3_dist = np.nan
          else:
            value_x = old_value_x
            value_y = old_value_y
            #To Right Eye
            L1_dist_x = coord_REyeX_r - value_x
            L1_dist_y = coord_REyeY_r - value_y
            L1_dist = math.sqrt(pow(L1_dist_x,2) + pow(L1_dist_y,2))
            #To Left Eye
            L2_dist_x = coord_LEyeX_r - value_x
            L2_dist_y = coord_LEyeY_r - value_y
            L2_dist = math.sqrt(pow(L2_dist_x,2) + pow(L2_dist_y,2))
            #To Nose
            L3_dist_x = coord_NoseX_r - value_x
            L3_dist_y = coord_NoseY_r - value_y
            L3_dist = math.sqrt(pow(L3_dist_x,2) + pow(L3_dist_y,2))
        else:
          #To Right Eye
          L1_dist_x = coord_REyeX_r - value_x
          L1_dist_y = coord_REyeY_r - value_y
          L1_dist = math.sqrt(pow(L1_dist_x,2) + pow(L1_dist_y,2))
          #To Left Eye
          L2_dist_x = coord_LEyeX_r - value_x
          L2_dist_y = coord_LEyeY_r - value_y
          L2_dist = math.sqrt(pow(L2_dist_x,2) + pow(L2_dist_y,2))
          #To Nose
          L3_dist_x = coord_NoseX_r - value_x
          L3_dist_y = coord_NoseY_r - value_y
          L3_dist = math.sqrt(pow(L3_dist_x,2) + pow(L3_dist_y,2))
        row_data.append(L1_dist)
        row_data.append(L1_dist_x)
        row_data.append(L1_dist_y)
        row_data.append(L2_dist)
        row_data.append(L2_dist_x)
        row_data.append(L2_dist_y)
        row_data.append(L3_dist)
        row_data.append(L3_dist_x)
        row_data.append(L3_dist_y)
    id_c += 1

  row_data.append(ref_dist)
  
  #Add targets to data
  found_target = False
  for i in data_target:
    if i[1].replace(" ", "") == image_name:
      row_data.append(i[2])
      found_target = True
 
  with open('data_output.csv','a') as fd:
    write = csv.writer(fd)   
    write.writerow(row_data)

  data.append(row_data)
  print(row_data)

  #Clear Old Values
  id_c = 0
  for n in results_hand_locations_list:
    if not np.isnan(results_hand_locations[id_c]):
      old_hand_values[n] = results_hand_locations[id_c]
    id_c += 1
  old_video_name = video_name
  frame_video_num +=1
  coordX_old = coordX
  coordY_old = coordY
  size_old = size
  values_old_reye = values_REye_r
  values_old_leye = values_LEye_r
  values_old_nose = values_Nose_r
  values_old_neck = values_Neck
  values_old_hip = values_Hip
  values_old_lwrist = values_LWrist_r
  values_old_rwrist = values_RWrist_r
  face_coords_final_old = face_coords_final
  frame_num_old = frame_num

print(data)