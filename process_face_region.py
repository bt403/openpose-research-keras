import cv2 as cv
import face_alignment
from skimage import io
from PIL import Image
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--processed_features', type=str, default='./data/estimates/processed_features.pkl')
parser.add_argument('--exported_frames', type=str, default='/content/drive/MyDrive/ResearchProject/exported_frames/')
parser.add_argument('--eyes_path', type=str, default='/content/drive/MyDrive/ResearchProject/eyes/')
parser.add_argument('--mouth_path', type=str, default='/content/drive/MyDrive/ResearchProject/mouth/')
args = parser.parse_args()

pdata = pd.read_pickle(args.processed_features)
exported_frames = args.exported_frames
eyes_path = args.eyes_path
mouth_path = args.mouth_path

def get_position(size, padding=0.6):
    
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                    0.553364, 0.490127, 0.42689]
    
    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                    0.784792, 0.824182, 0.831803, 0.824182]
    
    x, y = np.array(x), np.array(y)
    
    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    return np.array(list(zip(x, y)))

scores_array = []
scores_mouth_array = []
scores_eyes_array = []

coordX_start_array = []
coordX_end_array = []
coordY_start_array = []
coordY_end_array = []

eyes_x_min = []
eyes_x_max = []
eyes_y_min = []
eyes_y_max = []

mouth_x_min = []
mouth_x_max = []
mouth_y_min = []
mouth_y_max = []

old_video_name = None

for i, row in pdata.iterrows():
  path_cropped = row["path_cropped"]
  orientation = str(row["orientation"])
  orientation_img = str(path_cropped.split("/")[-1].split(".")[0].split("_")[-1])
  frame_name = path_cropped.split("/")[-1].replace("cropped_", "").replace("_" + str(orientation_img) + ".jpg", ".jpg")
  path = exported_frames + frame_name
  video_name = frame_name.split("_")[0] + "_" + frame_name.split("_")[1]
  coordX = row["coordX"]
  coordY = row["coordY"]
  size = row["size"]

  img_orig = io.imread(path)
  padding = 0.3
  if (orientation == "0"):
    coordY_rotated, coordX_rotated = coordY, coordX
    coordX_new = coordX - size*padding
    coordY_new = coordY - size*padding
    size = size*(1.0 + 2*padding)
  if (orientation == "270"):
    coordX_rotated = img_orig.shape[0] - coordY - size 
    coordX_new = coordX_rotated - size*padding
    coordY_rotated = coordX
    coordY_new = coordY_rotated - size*padding
    size = size*(1.0 + 2*padding)
    img_orig = np.rot90(img_orig, 3)

  if (orientation == "90"):
    coordX_rotated = coordY
    coordX_new = coordX_rotated - size*padding
    coordY_rotated = img_orig.shape[1] - coordX - size
    coordY_new = coordY_rotated - size*padding
    size = size*(1.0 + 2*padding)
    img_orig = np.rot90(img_orig)

  coordY_new_start = max(int(coordY_new),0)
  coordY_new_end = min(int(coordY_new+size), img_orig.shape[0])
  coordX_new_start = max(int(coordX_new),0)
  coordX_new_end = min(int(coordX_new+size), img_orig.shape[1])

  coordX_start_array.append(coordX_new_start)
  coordX_end_array.append(coordX_new_end)
  coordY_start_array.append(coordY_new_start)
  coordY_end_array.append(coordY_new_end)

  img = img_orig[coordY_new_start:coordY_new_end, coordX_new_start:coordX_new_end]

  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda', face_detector='sfd')
  (point, scores, _) = fa.get_landmarks(img, return_landmark_score=True) 

  if (video_name != old_video_name):
      coordX_min_eye = None
      coordX_max_eye = None
      coordY_min_eye = None
      coordY_max_eye = None
      coordX_min_mouth = None
      coordX_max_mouth = None
      coordY_min_mouth = None
      coordY_max_mouth = None

  front256 = get_position(img.shape[0])
  if(point is not None):
      score = np.mean(scores)
      scores_array.append(score)

      scores_filtered = scores[0][17:]
      scores_mouth = sum(scores_filtered[19:31])/len(scores_filtered[19:31])
      scores_eyes = sum(scores_filtered[31:])/len(scores_filtered[31:])

      scores_mouth_array.append(scores_mouth)
      scores_eyes_array.append(scores_eyes)

      shape = np.array(point[0])
      shape = shape
      shape = shape[17:,:2]

      ######
      # Eye Area
      ######
      #19-31
      coordX_min_eye = min(np.concatenate((shape[19:31,0], shape[0:13,0])))
      coordX_max_eye = max(np.concatenate((shape[19:31,0], shape[0:13,0])))
      coordY_min_eye = min(np.concatenate((shape[19:31,1], shape[0:13,1])))
      coordY_max_eye = max(np.concatenate((shape[19:31,1], shape[0:13,1])))
      
      sizeY_eye = coordY_max_eye - coordY_min_eye
      sizeX_eye = coordX_max_eye - coordX_min_eye

      coordX_min_eye = coordX_min_eye - 0.15*sizeX_eye
      coordX_max_eye = coordX_max_eye + 0.15*sizeX_eye
      sizeX_eye = sizeX_eye*1.3

      coordY_min_eye = coordY_min_eye - 0.1*sizeY_eye
      coordY_max_eye = coordY_max_eye + 0.1*sizeY_eye
      sizeY_eye = sizeY_eye*1.2

      if sizeY_eye < sizeX_eye/2:
        coordY_min_eye = coordY_min_eye - (sizeX_eye/2 - sizeY_eye)/2
        sizeY_eye = sizeX_eye/2
        coordY_max_eye = coordY_min_eye + sizeY_eye
      else:
        coordX_min_eye = coordX_min_eye - (sizeY_eye*2-sizeX_eye)/2
        sizeX_eye = sizeY_eye*2
        coordX_max_eye = coordX_min_eye + sizeX_eye

      eyes_x_min.append(coordX_min_eye)
      eyes_x_max.append(coordX_max_eye)
      eyes_y_min.append(coordY_min_eye)
      eyes_y_max.append(coordY_max_eye)  

      img_2 = img_orig[max(int(coordY_new_start + coordY_min_eye),0):min(int(coordY_new_start + coordY_max_eye), img_orig.shape[0]), max(int(coordX_new_start + coordX_min_eye),0):min(int(coordX_new_start + coordX_max_eye),img_orig.shape[1])]
      im = Image.fromarray(img_2)
      im = im.resize((160,80))
      print("saving")
      im.save(eyes_path + frame_name)

      ######
      # Mouth Area
      ######
      # >= 31 

      coordX_min_mouth = min(np.append(shape[31:,0],shape[17,0]))
      coordX_max_mouth = max(np.append(shape[31:,0],shape[17,0]))
      coordY_min_mouth = min(np.append(shape[31:,1],shape[17,1]))
      coordY_max_mouth = max(np.append(shape[31:,1],shape[17,1]))

      sizeY_mouth = coordY_max_mouth - coordY_min_mouth
      sizeX_mouth = coordX_max_mouth - coordX_min_mouth

      coordX_min_mouth = coordX_min_mouth - 0.15*sizeX_mouth
      coordX_max_mouth = coordX_max_mouth + 0.15*sizeX_mouth
      sizeX_mouth = sizeX_mouth*1.3

      coordY_min_mouth = coordY_min_mouth - 0.1*sizeY_mouth
      coordY_max_mouth = coordY_max_mouth + 0.1*sizeY_mouth
      sizeY_mouth = sizeY_mouth*1.2

      if sizeY_mouth < sizeX_mouth/2:
        coordY_min_mouth = coordY_min_mouth - (sizeX_mouth/2 - sizeY_mouth)/2
        sizeY_mouth = sizeX_mouth/2
        coordY_max_mouth = coordY_min_mouth + sizeY_mouth
      else:
        coordX_min_mouth = coordX_min_mouth - (sizeY_mouth*2-sizeX_mouth)/2
        sizeX_mouth = sizeY_mouth*2
        coordX_max_mouth = coordX_min_mouth + sizeX_mouth
      
      mouth_x_min.append(coordX_min_mouth)
      mouth_x_max.append(coordX_max_mouth)
      mouth_y_min.append(coordY_min_mouth)
      mouth_y_max.append(coordY_max_mouth)  

      img_2 = img_orig[max(int(coordY_new_start + coordY_min_mouth),0):min(int(coordY_new_start + coordY_max_mouth), img_orig.shape[0]), max(int(coordX_new_start + coordX_min_mouth),0):min(int(coordX_new_start + coordX_max_mouth),img_orig.shape[1])]
      im = Image.fromarray(img_2)
      im = im.resize((160,80))
      im.save(mouth_path + frame_name)

  else:
    scores_array.append(0)
    scores_mouth_array.append(0)
    scores_eyes_array.append(0)

    if video_name == old_video_name and coordX_max_eye is not None:
      eyes_x_min.append(coordX_min_eye)
      eyes_x_max.append(coordX_max_eye)
      eyes_y_min.append(coordY_min_eye)
      eyes_y_max.append(coordY_max_eye)  

      img_2 = img_orig[max(int(coordY_new_start + coordY_min_eye),0):min(int(coordY_new_start + coordY_max_eye), img_orig.shape[0]), max(int(coordX_new_start + coordX_min_eye),0):min(int(coordX_new_start + coordX_max_eye),img_orig.shape[1])]
      im = Image.fromarray(img_2)
      im = im.resize((160,80))
      im.save(eyes_path + frame_name)

      mouth_x_min.append(coordX_min_mouth)
      mouth_x_max.append(coordX_max_mouth)
      mouth_y_min.append(coordY_min_mouth)
      mouth_y_max.append(coordY_max_mouth) 
      
      img_2 = img_orig[max(int(coordY_new_start + coordY_min_mouth),0):min(int(coordY_new_start + coordY_max_mouth), img_orig.shape[0]), max(int(coordX_new_start + coordX_min_mouth),0):min(int(coordX_new_start + coordX_max_mouth),img_orig.shape[1])]
      im = Image.fromarray(img_2)
      im = im.resize((160,80))
      im.save(mouth_path + frame_name)
    else:
      print("entering here")
      size = row["size"]
      eyes_x_min.append(coordX_rotated)
      eyes_x_max.append(coordX_rotated+size)
      eyes_y_min.append(coordY_rotated)
      eyes_y_max.append(coordY_rotated+size/2)

      img_2 = img_orig[max(int(coordY_rotated),0):min(int(coordY_rotated + size/2), img_orig.shape[0]), max(int(coordX_rotated),0):min(int(coordX_rotated + size),img_orig.shape[1])]
      im = Image.fromarray(img_2)
      im = im.resize((160,80))
      im.save(eyes_path + frame_name)

      mouth_x_min.append(coordX_rotated)
      mouth_x_max.append(coordX_rotated+size)
      mouth_y_min.append(coordY_rotated+size/2)
      mouth_y_max.append(coordY_rotated+size)

      img_2 = img_orig[max(int(coordY_rotated+size/2),0):min(int(coordY_rotated + size), img_orig.shape[0]), max(int(coordX_rotated),0):min(int(coordX_rotated + size),img_orig.shape[1])]
      im = Image.fromarray(img_2)
      im = im.resize((160,80))
      im.save(mouth_path + frame_name)
  old_video_name = video_name

pdata["score_face_region"] = scores_array
pdata["score_eyes_region"] = scores_eyes_array
pdata["score_mouth_region"] = scores_mouth_array

pdata.to_pickle(args.processed_features)

