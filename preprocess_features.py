import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--processed_features', type=str, default='./data/estimates/processed_features.pkl')
parser.add_argument('--final_features', type=str, default='./data/estimates/final_features.pkl')
parser.add_argument('--final_features_augmented', type=str, default='./data/estimates/final_features_augmented.pkl')

args = parser.parse_args()

pdata = pd.read_pickle(args.processed_features)

columns = list(pdata.columns)
columns_filter = [x for x in columns if x not in ["video_name", "coordX", "coordY", "size", "orientation", "path", "path_cropped", "target", "ref_dist", "ears", "neck", "mouth", "cheeks", "eyes", "nose", "forehead"]]
columns_filter_without_angle = [x for x in columns_filter if x not in ["angle_LElbow", "angle_RElbow", "angle_LShoulder", "angle_RShoulder"]]
columns_filter_without_bool = [x for x in columns_filter if x not in ["coord_REye_Exists", "coord_LEye_Exists", "coord_REar_Exists", "coord_LEar_Exists", "coord_Nose_Exists"]]
columns_with_target = columns
columns_hand = []
id_c = 0
results_hand_locations_list = ["hand_found_R", "hand_found_L", "origin_x_hand_L", "origin_y_hand_L", "origin_x_hand_R", "origin_y_hand_R",
          "thumb_finger_x_hand_L", "thumb_finger_y_hand_L", "thumb_finger_x_hand_R", "thumb_finger_y_hand_R",
  "index_finger_x_hand_L", "index_finger_y_hand_L", "index_finger_x_hand_R", "index_finger_y_hand_R",
 "middle_finger_x_hand_L", "middle_finger_y_hand_L", "middle_finger_x_hand_R", "middle_finger_y_hand_R",
 "ring_finger_x_hand_L", "ring_finger_y_hand_L", "ring_finger_x_hand_R", "ring_finger_y_hand_R", 
 "pinky_finger_x_hand_L", "pinky_finger_y_hand_L", "pinky_finger_x_hand_R", "pinky_finger_y_hand_R"]
for i in results_hand_locations_list:
  if id_c < 2:
    columns.append(i)
  else:
    if (id_c%2 == 0): #x coordinate
      i2 = results_hand_locations_list[id_c+1]
      #To Right Eye
      L1_dist_x = "Lcoord_REyeX_" + i
      L1_dist_y = "Lcoord_REyeY_" + i2
      L1_dist = "Lcoord_REye_" + i.replace("_x", "").replace("_y", "")
      #To Left Eye
      L2_dist_x = "Lcoord_LEyeX_" + i
      L2_dist_y = "Lcoord_LEyeY_" + i2
      L2_dist = "Lcoord_LEye_" + i.replace("_x", "").replace("_y", "")
      #To Nose
      L3_dist_x = "Lcoord_NoseX_" + i
      L3_dist_y = "Lcoord_NoseY_" + i2
      L3_dist = "Lcoord_Nose_" + i.replace("_x", "").replace("_y", "")
      columns_hand.append(L1_dist_x)
      columns_hand.append(L1_dist_y)
      columns_hand.append(L1_dist)
      columns_hand.append(L2_dist_x)
      columns_hand.append(L2_dist_y)
      columns_hand.append(L2_dist)
      columns_hand.append(L3_dist_x)
      columns_hand.append(L3_dist_y)
      columns_hand.append(L3_dist)
  id_c += 1

def replace(group, stds):
    group[np.abs(group - group.mean()) > stds * group.std()] = np.nan
    return group

def df_column_switch(df, column1, column2, negative=False):
  df = df.rename(columns={column1:column2,column2:column1})
  if (negative):
    for i in df.columns:
      print(i)
    print(column1)
    df[column1] = df[column1]*(-1)
    df[column2] = df[column2]*(-1)
  return df

def augment_train(train):
  train_augmented = train.copy()
  train_augmented = df_column_switch(train_augmented,"L1_dist", "L2_dist")
  train_augmented = df_column_switch(train_augmented,"L1_dist_y", "L2_dist_y")
  train_augmented = df_column_switch(train_augmented,"L1_dist_x", "L2_dist_x", True)
  train_augmented = df_column_switch(train_augmented,"L3_dist", "L4_dist")
  train_augmented = df_column_switch(train_augmented,"L3_dist_y", "L4_dist_y")
  train_augmented = df_column_switch(train_augmented,"L3_dist_x", "L4_dist_x", True)
  train_augmented = df_column_switch(train_augmented,"L5_dist", "L6_dist")
  train_augmented = df_column_switch(train_augmented,"L5_dist_y", "L6_dist_y")
  train_augmented = df_column_switch(train_augmented,"L5_dist_x", "L6_dist_x", True)
  train_augmented = df_column_switch(train_augmented,"L7_dist", "L8_dist")
  train_augmented = df_column_switch(train_augmented,"L7_dist_y", "L8_dist_y")
  train_augmented = df_column_switch(train_augmented,"L7_dist_x", "L8_dist_x", True)
  train_augmented = df_column_switch(train_augmented,"L9_dist", "L12_dist")
  train_augmented = df_column_switch(train_augmented,"L9_dist_y", "L12_dist_y")
  train_augmented = df_column_switch(train_augmented,"L9_dist_x", "L12_dist_x", True)
  train_augmented = df_column_switch(train_augmented,"L10_dist", "L11_dist")
  train_augmented = df_column_switch(train_augmented,"L10_dist_y", "L11_dist_y")
  train_augmented = df_column_switch(train_augmented,"L10_dist_x", "L11_dist_x", True)
  train_augmented = df_column_switch(train_augmented,"displacement_RWrist", "displacement_LWrist")
  train_augmented = df_column_switch(train_augmented,"displacement_RElbow", "displacement_LElbow")
  train_augmented = df_column_switch(train_augmented,"speed_RElbow", "speed_LElbow")
  train_augmented = df_column_switch(train_augmented,"speed_RWrist", "speed_LWrist")
  train_augmented = df_column_switch(train_augmented,"sumAnglesL", "sumAnglesR")
  train_augmented = df_column_switch(train_augmented,"angle_RElbow", "angle_LElbow")
  train_augmented = df_column_switch(train_augmented,"angle_RShoulder", "angle_LShoulder")
  train_augmented = df_column_switch(train_augmented,"acceleration_RWrist_y","acceleration_LWrist_y")
  train_augmented = df_column_switch(train_augmented,"acceleration_RWrist_x","acceleration_LWrist_x")
  train_augmented = df_column_switch(train_augmented,"coord_REye_Exists", "coord_LEye_Exists")
  train_augmented = df_column_switch(train_augmented,"coord_REar_Exists", "coord_LEar_Exists")
  train_augmented = df_column_switch(train_augmented,"hand_found_L", "hand_found_R")
  train_augmented = df_column_switch(train_augmented,"distance_hand_L", "distance_hand_R")
  train_augmented = df_column_switch(train_augmented,"distance_x_hand_L", "distance_x_hand_R")
  train_augmented = df_column_switch(train_augmented,"distance_y_hand_L", "distance_y_hand_R")
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeX_origin_x_hand_L','Lcoord_LEyeX_origin_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeY_origin_y_hand_L','Lcoord_LEyeY_origin_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REye_origin_hand_L','Lcoord_LEye_origin_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeX_origin_x_hand_L','Lcoord_REyeX_origin_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeY_origin_y_hand_L','Lcoord_REyeY_origin_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEye_origin_hand_L','Lcoord_REye_origin_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseX_origin_x_hand_L','Lcoord_NoseX_origin_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseY_origin_y_hand_L','Lcoord_NoseY_origin_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_Nose_origin_hand_L','Lcoord_Nose_origin_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeX_thumb_finger_x_hand_L','Lcoord_LEyeX_thumb_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeY_thumb_finger_y_hand_L','Lcoord_LEyeY_thumb_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REye_thumb_finger_hand_L','Lcoord_LEye_thumb_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeX_thumb_finger_x_hand_L','Lcoord_REyeX_thumb_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeY_thumb_finger_y_hand_L','Lcoord_REyeY_thumb_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEye_thumb_finger_hand_L','Lcoord_REye_thumb_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseX_thumb_finger_x_hand_L','Lcoord_NoseX_thumb_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseY_thumb_finger_y_hand_L','Lcoord_NoseY_thumb_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_Nose_thumb_finger_hand_L','Lcoord_Nose_thumb_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeX_index_finger_x_hand_L','Lcoord_LEyeX_index_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeY_index_finger_y_hand_L','Lcoord_LEyeY_index_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REye_index_finger_hand_L','Lcoord_LEye_index_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeX_index_finger_x_hand_L','Lcoord_REyeX_index_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeY_index_finger_y_hand_L','Lcoord_REyeY_index_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEye_index_finger_hand_L','Lcoord_REye_index_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseX_index_finger_x_hand_L','Lcoord_NoseX_index_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseY_index_finger_y_hand_L','Lcoord_NoseY_index_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_Nose_index_finger_hand_L','Lcoord_Nose_index_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeX_middle_finger_x_hand_L','Lcoord_LEyeX_middle_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeY_middle_finger_y_hand_L','Lcoord_LEyeY_middle_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REye_middle_finger_hand_L','Lcoord_LEye_middle_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeX_middle_finger_x_hand_L','Lcoord_REyeX_middle_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeY_middle_finger_y_hand_L','Lcoord_REyeY_middle_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEye_middle_finger_hand_L','Lcoord_REye_middle_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseX_middle_finger_x_hand_L','Lcoord_NoseX_middle_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseY_middle_finger_y_hand_L','Lcoord_NoseY_middle_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_Nose_middle_finger_hand_L','Lcoord_Nose_middle_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeX_ring_finger_x_hand_L','Lcoord_LEyeX_ring_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeY_ring_finger_y_hand_L','Lcoord_LEyeY_ring_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REye_ring_finger_hand_L','Lcoord_LEye_ring_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeX_ring_finger_x_hand_L','Lcoord_REyeX_ring_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeY_ring_finger_y_hand_L','Lcoord_REyeY_ring_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEye_ring_finger_hand_L','Lcoord_REye_ring_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseX_ring_finger_x_hand_L','Lcoord_NoseX_ring_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseY_ring_finger_y_hand_L','Lcoord_NoseY_ring_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_Nose_ring_finger_hand_L','Lcoord_Nose_ring_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeX_pinky_finger_x_hand_L','Lcoord_LEyeX_pinky_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_REyeY_pinky_finger_y_hand_L','Lcoord_LEyeY_pinky_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_REye_pinky_finger_hand_L','Lcoord_LEye_pinky_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeX_pinky_finger_x_hand_L','Lcoord_REyeX_pinky_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEyeY_pinky_finger_y_hand_L','Lcoord_REyeY_pinky_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_LEye_pinky_finger_hand_L','Lcoord_REye_pinky_finger_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseX_pinky_finger_x_hand_L','Lcoord_NoseX_pinky_finger_x_hand_R', True)
  train_augmented = df_column_switch(train_augmented,'Lcoord_NoseY_pinky_finger_y_hand_L','Lcoord_NoseY_pinky_finger_y_hand_R')
  train_augmented = df_column_switch(train_augmented,'Lcoord_Nose_pinky_finger_hand_L','Lcoord_Nose_pinky_finger_hand_R')
  return train_augmented

def df_row_switch(rw, column1, column2, negativeX=False, negativeY=False):
  rw = rw.copy()
  temp = rw[column1]
  rw[column1] = rw[column2]
  rw[column2] = temp
  if (negativeX):
    rw[column1] = rw[column1]*(-1)
  if (negativeY):
    rw[column2] = rw[column2]*(-1)
  return rw

'''
270° -> DistX = -DistY —> DistY = DistX
——> Dist Y = -DistX -> DistX = DistY
Dist X = -3 - (-5) = 2
Dist Y = 2- 1 = 1
'''

def switch_x_y_orientation(train):
  train_augmented = train.copy()
  for i, row in train_augmented.iterrows():
    if row["orientation"] == 270:
      row = df_row_switch(row,"L3_dist_x", "L3_dist_y", True)
      row = df_row_switch(row,"L4_dist_x", "L4_dist_y", True)
      
      row = df_row_switch(row,'Lcoord_REyeX_origin_x_hand_L','Lcoord_REyeY_origin_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_REyeX_origin_x_hand_R','Lcoord_REyeY_origin_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_LEyeX_origin_x_hand_L','Lcoord_LEyeY_origin_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_LEyeX_origin_x_hand_R','Lcoord_LEyeY_origin_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_NoseX_origin_x_hand_L','Lcoord_NoseY_origin_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_NoseX_origin_x_hand_R','Lcoord_NoseY_origin_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_REyeX_thumb_finger_x_hand_L','Lcoord_REyeY_thumb_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_REyeX_thumb_finger_x_hand_R','Lcoord_REyeY_thumb_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_LEyeX_thumb_finger_x_hand_L','Lcoord_LEyeY_thumb_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_LEyeX_thumb_finger_x_hand_R','Lcoord_LEyeY_thumb_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_NoseX_thumb_finger_x_hand_L','Lcoord_NoseY_thumb_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_NoseX_thumb_finger_x_hand_R','Lcoord_NoseY_thumb_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_REyeX_index_finger_x_hand_L','Lcoord_REyeY_index_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_REyeX_index_finger_x_hand_R','Lcoord_REyeY_index_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_LEyeX_index_finger_x_hand_L','Lcoord_LEyeY_index_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_LEyeX_index_finger_x_hand_R','Lcoord_LEyeY_index_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_NoseX_index_finger_x_hand_L','Lcoord_NoseY_index_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_NoseX_index_finger_x_hand_R','Lcoord_NoseY_index_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_REyeX_middle_finger_x_hand_L','Lcoord_REyeY_middle_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_REyeX_middle_finger_x_hand_R','Lcoord_REyeY_middle_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_LEyeX_middle_finger_x_hand_L','Lcoord_LEyeY_middle_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_LEyeX_middle_finger_x_hand_R','Lcoord_LEyeY_middle_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_NoseX_middle_finger_x_hand_L','Lcoord_NoseY_middle_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_NoseX_middle_finger_x_hand_R','Lcoord_NoseY_middle_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_REyeX_ring_finger_x_hand_L','Lcoord_REyeY_ring_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_REyeX_ring_finger_x_hand_R','Lcoord_REyeY_ring_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_LEyeX_ring_finger_x_hand_L','Lcoord_LEyeY_ring_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_LEyeX_ring_finger_x_hand_R','Lcoord_LEyeY_ring_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_NoseX_ring_finger_x_hand_L','Lcoord_NoseY_ring_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_NoseX_ring_finger_x_hand_R','Lcoord_NoseY_ring_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_REyeX_pinky_finger_x_hand_L','Lcoord_REyeY_pinky_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_REyeX_pinky_finger_x_hand_R','Lcoord_REyeY_pinky_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_LEyeX_pinky_finger_x_hand_L','Lcoord_LEyeY_pinky_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_LEyeX_pinky_finger_x_hand_R','Lcoord_LEyeY_pinky_finger_y_hand_R', True)

      row = df_row_switch(row,'Lcoord_NoseX_pinky_finger_x_hand_L','Lcoord_NoseY_pinky_finger_y_hand_L', True)
      row = df_row_switch(row,'Lcoord_NoseX_pinky_finger_x_hand_R','Lcoord_NoseY_pinky_finger_y_hand_R', True)
      train_augmented.loc[i] = row
    if row["orientation"] == 90:
      row = df_row_switch(row,"L3_dist_x", "L3_dist_y", False, True)
      row = df_row_switch(row,"L4_dist_x", "L4_dist_y", False, True)
      
      row = df_row_switch(row,'Lcoord_REyeX_origin_x_hand_L','Lcoord_REyeY_origin_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_REyeX_origin_x_hand_R','Lcoord_REyeY_origin_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_LEyeX_origin_x_hand_L','Lcoord_LEyeY_origin_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_LEyeX_origin_x_hand_R','Lcoord_LEyeY_origin_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_NoseX_origin_x_hand_L','Lcoord_NoseY_origin_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_NoseX_origin_x_hand_R','Lcoord_NoseY_origin_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_REyeX_thumb_finger_x_hand_L','Lcoord_REyeY_thumb_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_REyeX_thumb_finger_x_hand_R','Lcoord_REyeY_thumb_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_LEyeX_thumb_finger_x_hand_L','Lcoord_LEyeY_thumb_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_LEyeX_thumb_finger_x_hand_R','Lcoord_LEyeY_thumb_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_NoseX_thumb_finger_x_hand_L','Lcoord_NoseY_thumb_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_NoseX_thumb_finger_x_hand_R','Lcoord_NoseY_thumb_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_REyeX_index_finger_x_hand_L','Lcoord_REyeY_index_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_REyeX_index_finger_x_hand_R','Lcoord_REyeY_index_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_LEyeX_index_finger_x_hand_L','Lcoord_LEyeY_index_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_LEyeX_index_finger_x_hand_R','Lcoord_LEyeY_index_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_NoseX_index_finger_x_hand_L','Lcoord_NoseY_index_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_NoseX_index_finger_x_hand_R','Lcoord_NoseY_index_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_REyeX_middle_finger_x_hand_L','Lcoord_REyeY_middle_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_REyeX_middle_finger_x_hand_R','Lcoord_REyeY_middle_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_LEyeX_middle_finger_x_hand_L','Lcoord_LEyeY_middle_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_LEyeX_middle_finger_x_hand_R','Lcoord_LEyeY_middle_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_NoseX_middle_finger_x_hand_L','Lcoord_NoseY_middle_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_NoseX_middle_finger_x_hand_R','Lcoord_NoseY_middle_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_REyeX_ring_finger_x_hand_L','Lcoord_REyeY_ring_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_REyeX_ring_finger_x_hand_R','Lcoord_REyeY_ring_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_LEyeX_ring_finger_x_hand_L','Lcoord_LEyeY_ring_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_LEyeX_ring_finger_x_hand_R','Lcoord_LEyeY_ring_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_NoseX_ring_finger_x_hand_L','Lcoord_NoseY_ring_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_NoseX_ring_finger_x_hand_R','Lcoord_NoseY_ring_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_REyeX_pinky_finger_x_hand_L','Lcoord_REyeY_pinky_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_REyeX_pinky_finger_x_hand_R','Lcoord_REyeY_pinky_finger_y_hand_R', False, True)

      row = df_row_switch(row,'Lcoord_LEyeX_pinky_finger_x_hand_L','Lcoord_LEyeY_pinky_finger_y_hand_L', False, True)
      row = df_row_switch(row,'Lcoord_LEyeX_pinky_finger_x_hand_R','Lcoord_LEyeY_pinky_finger_y_hand_R', False, True)

      train_augmented.loc[i] = row
  return train_augmented

# Normalise features that were not normalised previously. Mainly the features of the hand.
pdata["ref_dist"] = pdata[["ref_dist", "video_name"]].groupby('video_name').apply(lambda group: group.interpolate(method='index'))["ref_dist"]
pdata["ref_dist"] = pdata[["ref_dist", "video_name"]].groupby('video_name').apply(lambda group: group.interpolate(method='bfill'))["ref_dist"]
pdata["ref_dist"] = pdata[["ref_dist", "video_name"]].groupby('video_name').apply(lambda group: group.interpolate(method='pad'))["ref_dist"]
pdata["ref_dist"] = pdata.groupby('video_name')['ref_dist'].transform(lambda s: s.rolling(5, min_periods=5).mean())
pdata.loc[:,"ref_dist"] = pdata.loc[:,"ref_dist"].fillna(pdata.groupby('video_name')["ref_dist"].transform('mean'))
pdata.loc[pdata["hand_found_R"].isna(),"hand_found_R"] = 0
pdata.loc[pdata["hand_found_L"].isna(),"hand_found_L"] = 0
pdata.loc[:, columns_hand] = pdata[columns_hand + ["video_name"]].groupby("video_name").transform(lambda g: replace(g, 1.8))
for i in columns_hand:
  print(i)
  pdata[i] = pdata[i]/pdata["ref_dist"]
for i in ["L3_dist", "L3_dist_x", "L3_dist_y", "L4_dist", "L4_dist_x", "L4_dist_y"]:
  pdata[i] = pdata[i]/pdata["ref_dist"]

#Replace outliers per video with NaN
pdata.loc[:, columns_filter] = pdata[columns_filter + ["video_name"]].groupby("video_name").transform(lambda g: replace(g, 3.5))

#Interpolate missing values per video
pdata = pdata.groupby('video_name').apply(lambda group: group.interpolate(method='index',limit=3))
pdata = pdata.groupby('video_name').apply(lambda group: group.interpolate(method='pad',limit=3))
pdata = pdata.groupby('video_name').apply(lambda group: group.interpolate(method='bfill',limit=3))

#Replace remaining missing values with mean per video
for i in columns_filter:
  pdata.loc[:,i] = pdata[i].fillna(pdata.groupby('video_name')[i].transform('mean'))

#Convert columns to int
pdata['coord_REye_Exists'] = pdata['coord_REye_Exists'].astype(int)
pdata['coord_LEye_Exists'] = pdata['coord_LEye_Exists'].astype(int)
pdata['coord_REar_Exists'] = pdata['coord_REar_Exists'].astype(int)
pdata['coord_LEar_Exists'] = pdata['coord_LEar_Exists'].astype(int)

#Generate augmented data by flipping values horizontally
pdata_augmented = augment_train(pdata)

#Swith x and y orientation of features that were not rotated and normalised before according to the orientation of the baby
pdata = switch_x_y_orientation(pdata)
pdata_augmented = switch_x_y_orientation(pdata_augmented)

pdata.to_pickle(args.final_features)
pdata_augmented.to_pickle(args.final_features_augmented)