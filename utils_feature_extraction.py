import pandas as pd
import math
import cv2
import numpy as np
from PIL import Image

def removeOutliers(group):
  c = 0
  print(group["video"].values[0])
  compare_stdX = 2.4*group.x.std()
  meanX = group.x.mean()
  compare_stdY = 2.4*group.y.std()
  meanY = group.y.mean()
  count = 0
  print(group["bp"].values[c])
  for i in group["x"].values:
    x_val = group["x"].values[c]
    x_val = np.abs(x_val-meanX)
    y_val = group["y"].values[c]
    y_val = np.abs(y_val-meanY)
    if x_val > compare_stdX:
        group["x"].values[c] = np.NaN
        group["y"].values[c] = np.NaN
        count+=1
    if y_val > compare_stdY:
        group["x"].values[c] = np.NaN
        group["y"].values[c] = np.NaN
        count+=1
    c+=1
  return group

def getBoxCoordinates(coord_HipX, coord_HipY, coord_NeckX, coord_NeckY, pixel_x, pixel_y, orientation):
  if (coord_HipX is None or coord_HipY is None or coord_NeckX is None or coord_NeckY is None):
    return (None, None, None)
  distX = abs(coord_HipX - coord_NeckX)
  distY = abs(coord_HipY - coord_NeckY)
  dist = math.sqrt(pow(distX,2) + pow(distY,2))
  size = dist*2
  if (size > pixel_y):
    size = pixel_y
  if (size > pixel_x):
    size = pixel_x
  orientation_new = 0
  if (distX > distY):
    #Horizontal Baby
    if (coord_HipX > coord_NeckX):
      orientation_new = 270
      if orientation_new == orientation:
        #Head to the Left, Hip to the Right
        coordX = coord_NeckX - dist*1.5
        coordY = coord_NeckY - dist
    else:
      orientation_new = 90
      if orientation_new == orientation:
        #Head to the Right, Hip to the Left
        coordX = coord_NeckX - dist*0.5
        coordY = coord_NeckY - dist
  else:
    #Vertical Baby
    if (coord_HipY > coord_NeckY):
      #Head on the top
      if orientation_new == orientation:
        coordY = coord_NeckY - dist*1.5
        coordX = coord_NeckX - dist
    else:
      #Head upside down
      if orientation_new == orientation:
        coordY = coord_NeckY - dist*0.5
        coordX = coord_NeckX - dist
  if orientation_new == orientation:
    if (coordX < 0):
      coordX = 0
    if (coordY < 0):
      coordY = 0
    if (coordX + size > int(pixel_x)):
      coordX = int(pixel_x) - size
    if (coordY + size > int(pixel_y)):
      coordY = int(pixel_y) - size
    return (coordX, coordY, size)
  else:
    return (None, None, None)

def addMinMax(df):
  df['min_x'] = df[x_columns].min(axis=1)
  df['max_x'] = df[x_columns].max(axis=1)
  df['min_y'] = df[y_columns].min(axis=1)
  df['max_y'] = df[y_columns].max(axis=1)
  return df

def getFaces(df, y_max, x_max, eyeLeft, eyeRight, neck, coordX, coordY, size, angle, limit=False, video_name=None, video_name_now=None):
  faces = []
  not_changed_video = True
  for index, f in df.iterrows():
      height = int(f["max_x"]) - int(f["min_x"])
      width = int(f["max_y"]) - int(f["min_y"])
      if (f["confidence"] > 0.1):
        if angle == 0:
          face_coords = (f["min_x"], f["min_y"], f["max_x"], f["max_y"], f["eye_lmk_x_1"], f["eye_lmk_y_1"], f["eye_lmk_x_31"], f["eye_lmk_y_31"], f["confidence"])
        elif angle == 270:
          face_coords = (f["min_y"], x_max - int(f["min_x"]) - width, f["max_y"], x_max - int(f["max_x"]) + width, f["eye_lmk_y_1"], x_max - int(f["eye_lmk_x_1"]), f["eye_lmk_y_31"], x_max - int(f["eye_lmk_x_31"]), f["confidence"])
        else:
          face_coords = (y_max - int(f["min_y"]) - height, int(f["min_x"]), y_max - int(f["max_y"]) + height, int(f["max_x"]), y_max - int(f["eye_lmk_y_1"]), f["eye_lmk_x_1"], y_max - int(f["eye_lmk_y_31"]), f["eye_lmk_x_31"], f["confidence"])
        if (limit):
          if (video_name_now != video_name):
            not_changed_video = False # check that faces are from the same video. If new video, stop searching
          elif (face_coords[0] >= coordX and face_coords[1] >= coordY and face_coords[2] < coordX + size and face_coords[3] < coordY + size):
            faces.append(face_coords)
        else:
          if (face_coords[0] >= coordX and face_coords[1] >= coordY and face_coords[2] < coordX + size and face_coords[3] < coordY + size):
            faces.append(face_coords)
  if (len(faces) == 0):
    if (not np.isnan(eyeLeft[0]) or not np.isnan(eyeRight[0])):
      if (np.isnan(eyeLeft[0])):
        print("-")
      if (np.isnan(eyeRight[0])):
        print("-")
      else:
        distX = abs(eyeLeft[0] - eyeRight[0])
        distY = abs(eyeLeft[1] - eyeRight[1])
        dist_eye = math.sqrt(pow(distX,2) + pow(distY,2))
        if (np.isnan(neck[0])):
          dist = dist_eye
        else:
          distX = abs(eyeLeft[0] - neck[0])
          distY = abs(eyeLeft[1] - neck[1])
          dist_neck = math.sqrt(pow(distX,2) + pow(distY,2))*0.5
          dist = max(dist_eye, dist_neck)
        if angle > 0:
          minX = min(eyeLeft[0], eyeRight[0]) - dist*0.5
          minY = min(eyeLeft[1], eyeRight[1]) - dist*0.5
          maxX = max(eyeLeft[0], eyeRight[0]) + dist*0.5
          maxY = max(eyeLeft[1], eyeRight[1]) + dist*0.5
        else:
          minX = min(eyeLeft[0], eyeRight[0]) - dist*0.5
          minY = min(eyeLeft[1], eyeRight[1]) - dist*0.5
          maxX = max(eyeLeft[0], eyeRight[0]) + dist*0.5
          maxY = max(eyeLeft[1], eyeRight[1]) + dist*0.5
        face_coords = (minX, minY, maxX, maxY, eyeRight[0], eyeRight[1], eyeLeft[0], eyeLeft[1], 1)
        faces.append(face_coords)
  else:
    print("-----")
  if limit:
    return (faces, not_changed_video)
  else:
    return faces

def getFaceCoordsFinal(faces, faces_90, faces_270, coord_NeckX, coord_NeckY, orientation, oldEyes = True):
  dist_min = 10000
  face_coords_final = face_coords_final_old = None
  faces_loop = None
  other_faces_loop = None

  if orientation == 0:
    faces_loop = faces
    other_faces_loop = faces_90.append(faces_270)
  elif orientation == 90:
    faces_loop = faces_90
    other_faces_loop = faces.append(faces_270)
  elif orientation == 270:
    faces_loop = faces_270
    other_faces_loop = faces.append(faces_90)
  
  for f in faces_loop:
    face_eye_coords = (f[4], f[5])
    distX = abs(face_eye_coords[0] - coord_NeckX)
    distY = abs(face_eye_coords[1] - coord_NeckY)
    dist = math.sqrt(pow(distX,2) + pow(distY,2))
    if dist < dist_min and f[8] > 0.02:
      if orientation == 90:
        if coord_NeckX < f[0]:
          face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
          dist_min = dist
      elif orientation == 270:
        if coord_NeckX > f[2]:
          face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
          dist_min = dist
      else:
        if coord_NeckY < f[1]:
          face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
          dist_min = dist
  if (not oldEyes):
    if (face_coords_final is None):
      if other_faces_loop is not None:
        for f in other_faces_loop:
          face_eye_coords = (f[4], f[5])
          distX = abs(face_eye_coords[0] - coord_NeckX)
          distY = abs(face_eye_coords[1] - coord_NeckY)
          dist = math.sqrt(pow(distX,2) + pow(distY,2))
          if dist < dist_min and f[8] > 0.02:
            if orientation == 90:
              if coord_NeckX < f[0]:
                face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
                dist_min = dist
            elif orientation == 270:
              if coord_NeckX > f[2]:
                face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
                dist_min = dist
            else:
              if coord_NeckY < f[1]:
                face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
                dist_min = dist
    if (face_coords_final is None):
      for f in faces_loop:
        face_eye_coords = (f[4], f[5])
        distX = abs(face_eye_coords[0] - coord_NeckX)
        distY = abs(face_eye_coords[1] - coord_NeckY)
        dist = math.sqrt(pow(distX,2) + pow(distY,2))
        if dist < dist_min and f[8] > 0.02:
          face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
          dist_min = dist
    if (face_coords_final is None):
      if other_faces_loop is not None:
        for f in other_faces_loop:
          face_eye_coords = (f[4], f[5])
          distX = abs(face_eye_coords[0] - coord_NeckX)
          distY = abs(face_eye_coords[1] - coord_NeckY)
          dist = math.sqrt(pow(distX,2) + pow(distY,2))
          if dist < dist_min and f[8] > 0.02:
            face_coords_final = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
            dist_min = dist
  return face_coords_final

def alignFace(rightEyeCenter, leftEyeCenter, neck, size, image_s, orientation, y_max, x_max):
  if orientation == 0:
    print("orientation 0")
  elif orientation == 270:
    print("orientation 270")
    rightEyeCenter = (y_max - int(rightEyeCenter[1]), int(rightEyeCenter[0]))
    leftEyeCenter = (y_max - int(leftEyeCenter[1]), int(leftEyeCenter[0]))
    neck = (y_max - int(neck[1]), int(neck[0]))
    image_s  = Image.fromarray(image_s).transpose(Image.ROTATE_270)
    image_s = np.array(image_s.convert('RGB'))
  else:
    print("orientation 90")
    rightEyeCenter = (rightEyeCenter[1], x_max - int(rightEyeCenter[0]))
    leftEyeCenter = (leftEyeCenter[1], x_max - int(leftEyeCenter[0]))
    neck = (neck[1], x_max - int(neck[0]))
    image_s  = Image.fromarray(image_s).transpose(Image.ROTATE_90)
    image_s = np.array(image_s.convert('RGB'))

  dY = rightEyeCenter[1] - leftEyeCenter[1]
  dX = rightEyeCenter[0] - leftEyeCenter[0]
  angle = np.degrees(np.arctan2(dY, dX))
  angle = angle - 180

  if ((angle > 45 and angle < 315) or (angle < -45 and angle > -315)):
    angle = 0

  desiredRightEyeX = 1.0 - 0.35
  dist_eye = np.sqrt((dX ** 2) + (dY ** 2))
  eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2), int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
  if (np.isnan(neck[0])):
    dist = dist_eye
  else:
    distX_l = abs(leftEyeCenter[0] - neck[0])
    distY_l = abs(leftEyeCenter[1] - neck[1])
    distX_r = abs(rightEyeCenter[0] - neck[0])
    distY_r = abs(rightEyeCenter[1] - neck[1])
    dist_neck_l = np.sqrt((distX_l ** 2) + (distY_l ** 2))*0.3
    dist_neck_r = np.sqrt((distX_r ** 2) + (distY_r ** 2))*0.3
    dist_neck = max(dist_neck_l, dist_neck_r)

    dist = max(dist_eye, dist_neck) 
    if (dist_neck == dist):
      eyesCenter = (int((leftEyeCenter[0] + neck[0]) // 2), int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
      distY = (distY_l + distY_r)/2.0
      if (dist_eye < 0.3*distY):
        dist = dist*1.6
    else:
      distY = (distY_l + distY_r)/2.0
      if (dist_eye > 3*distY):
        dist = dist/1.5

  desiredDist = (desiredRightEyeX - 0.35)
  desiredDist *= size
  scale = desiredDist / dist
 
  M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
  tX = size * 0.5
  tY = size * 0.35
  M[0, 2] += (tX - eyesCenter[0])
  M[1, 2] += (tY - eyesCenter[1])
  (w, h) = (int(size), int(size))

  output = cv2.warpAffine(image_s, M, (w, h), flags=cv2.INTER_CUBIC)
  return output
  
def getHandData(file_full, orientation, size, coordX, coordY, sizeX, sizeY, idx):     
  hand_found_R = np.nan
  hand_found_L = np.nan
  origin_x_hand_L = np.nan
  origin_y_hand_L = np.nan
  origin_x_hand_R = np.nan
  origin_y_hand_R = np.nan 
  
  thumb_finger_x_hand_L = np.nan
  thumb_finger_y_hand_L = np.nan
  thumb_finger_x_hand_R = np.nan
  thumb_finger_y_hand_R = np.nan

  index_finger_x_hand_L = np.nan
  index_finger_y_hand_L = np.nan
  index_finger_x_hand_R = np.nan
  index_finger_y_hand_R = np.nan

  middle_finger_x_hand_L = np.nan
  middle_finger_y_hand_L = np.nan
  middle_finger_x_hand_R = np.nan
  middle_finger_y_hand_R = np.nan

  ring_finger_x_hand_L = np.nan
  ring_finger_y_hand_L = np.nan
  ring_finger_x_hand_R = np.nan
  ring_finger_y_hand_R = np.nan

  pinky_finger_x_hand_L = np.nan
  pinky_finger_y_hand_L = np.nan
  pinky_finger_x_hand_R = np.nan
  pinky_finger_y_hand_R = np.nan

  img = Image.open(file_full)
  if sizeX is None or sizeY is None:
    if (orientation == 0):
      coordX_new = coordX - size*0.25
      coordY_new = coordY - size*0.1
      sizeX = size*1.5
      sizeY = size*1.6
    if (orientation == 270):
      coordX_new = coordX - size*0.1
      coordY_new = coordY - size*0.25
      sizeX = size*1.6
      sizeY = size*1.5
    if (orientation == 90):
      coordX_new = coordX - size*0.5
      coordY_new = coordY - size*0.25
      sizeX = size*1.6
      sizeY = size*1.5
  else:
    coordX_new = coordX
    coordY_new = coordY
  box = coordX_new, coordY_new, coordX_new+sizeX, coordY_new + sizeY
  img_2 = img.crop(box)
  img_2.save("temp + " + str(orientation) + ".jpg")

  img_cv2 = np.array(img_2.convert('RGB'))
  image = cv2.cvtColor(cv2.flip(img_cv2, 1), cv2.COLOR_RGB2BGR)
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  image_height, image_width, _ = image.shape
  annotated_image = image.copy()

  if not results.multi_hand_landmarks:
    return (hand_found_R, hand_found_L, origin_x_hand_L, origin_y_hand_L, origin_x_hand_R, origin_y_hand_R,
              thumb_finger_x_hand_L, thumb_finger_y_hand_L, thumb_finger_x_hand_R, thumb_finger_y_hand_R,
      index_finger_x_hand_L, index_finger_y_hand_L, index_finger_x_hand_R, index_finger_y_hand_R,
      middle_finger_x_hand_L, middle_finger_y_hand_L, middle_finger_x_hand_R, middle_finger_y_hand_R,
      ring_finger_x_hand_L, ring_finger_y_hand_L, ring_finger_x_hand_R, ring_finger_y_hand_R,
      pinky_finger_x_hand_L, pinky_finger_y_hand_L, pinky_finger_x_hand_R, pinky_finger_y_hand_R)
  else:
    found_left = False
    found_right = False
    for idx_r, hand_handedness in enumerate(results.multi_handedness):
      score = hand_handedness.classification[0].score
      hand_landmarks = results.multi_hand_landmarks[idx_r]
      count_landmark = 0.0
      total_landmark_x = 0
      total_landmark_y = 0
      for landmark in hand_landmarks.landmark:
        total_landmark_x += landmark.x
        total_landmark_y += landmark.y
        count_landmark += 1
      
      thumb_finger_x = hand_landmarks.landmark[4].x*sizeX + coordX_new
      thumb_finger_y = hand_landmarks.landmark[4].y*sizeY + coordY_new
      index_finger_x = hand_landmarks.landmark[8].x*sizeX + coordX_new
      index_finger_y = hand_landmarks.landmark[8].y*sizeY + coordY_new
      middle_finger_x = hand_landmarks.landmark[12].x*sizeX + coordX_new
      middle_finger_y = hand_landmarks.landmark[12].y*sizeY + coordY_new
      ring_finger_x = hand_landmarks.landmark[16].x*sizeX + coordX_new
      ring_finger_y = hand_landmarks.landmark[16].y*sizeY + coordY_new
      pinky_finger_x = hand_landmarks.landmark[20].x*sizeX + coordX_new
      pinky_finger_y = hand_landmarks.landmark[20].y*sizeY + coordY_new

      origin_x = total_landmark_x/count_landmark*sizeX + coordX_new
      origin_y = total_landmark_y/count_landmark*sizeY + coordY_new

      #distance = math.sqrt(origin_x*origin_x + origin_y*origin_y) 

      if hand_handedness.classification[0].label == "Left":
        if found_left:
          #column_distance_hand_R = distance
          origin_x_hand_R = origin_x
          origin_y_hand_R = origin_y
          thumb_finger_x_hand_R = thumb_finger_x
          thumb_finger_y_hand_R = thumb_finger_y
          index_finger_x_hand_R = index_finger_x
          index_finger_y_hand_R = index_finger_y
          middle_finger_x_hand_R = middle_finger_x
          middle_finger_y_hand_R = middle_finger_y
          ring_finger_x_hand_R = ring_finger_x
          ring_finger_y_hand_R = ring_finger_y
          pinky_finger_x_hand_R = pinky_finger_x
          pinky_finger_y_hand_R = pinky_finger_y
          hand_found_R = score
          found_right = True
        else:
          #column_distance_hand_L = distance
          origin_x_hand_L = origin_x
          origin_y_hand_L = origin_y
          thumb_finger_x_hand_L = thumb_finger_x
          thumb_finger_y_hand_L = thumb_finger_y
          index_finger_x_hand_L = index_finger_x
          index_finger_y_hand_L = index_finger_y
          middle_finger_x_hand_L = middle_finger_x
          middle_finger_y_hand_L = middle_finger_y
          ring_finger_x_hand_L = ring_finger_x
          ring_finger_y_hand_L = ring_finger_y
          pinky_finger_x_hand_L = pinky_finger_x
          pinky_finger_y_hand_L = pinky_finger_y
          hand_found_L = score
          found_left = True
      if hand_handedness.classification[0].label == "Right":
        if found_right:
          #column_distance_hand_L = distance
          origin_x_hand_L = origin_x
          origin_y_hand_L = origin_y
          thumb_finger_x_hand_L = thumb_finger_x
          thumb_finger_y_hand_L = thumb_finger_y
          index_finger_x_hand_L = index_finger_x
          index_finger_y_hand_L = index_finger_y
          middle_finger_x_hand_L = middle_finger_x
          middle_finger_y_hand_L = middle_finger_y
          ring_finger_x_hand_L = ring_finger_x
          ring_finger_y_hand_L = ring_finger_y
          pinky_finger_x_hand_L = pinky_finger_x
          pinky_finger_y_hand_L = pinky_finger_y
          hand_found_L = score
          found_left = True
        else:
          #column_distance_hand_R = distance
          origin_x_hand_R = origin_x
          origin_y_hand_R = origin_y
          thumb_finger_x_hand_R = thumb_finger_x
          thumb_finger_y_hand_R = thumb_finger_y
          index_finger_x_hand_R = index_finger_x
          index_finger_y_hand_R = index_finger_y
          middle_finger_x_hand_R = middle_finger_x
          middle_finger_y_hand_R = middle_finger_y
          ring_finger_x_hand_R = ring_finger_x
          ring_finger_y_hand_R = ring_finger_y
          pinky_finger_x_hand_R = pinky_finger_x
          pinky_finger_y_hand_R = pinky_finger_y
          hand_found_R = score
          found_right = True

      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
  cv2.imwrite(
      '/content/openpose-research-keras/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
  # Draw hand world landmarks.
  if not results.multi_hand_world_landmarks:
    return (hand_found_R, hand_found_L, origin_x_hand_L, origin_y_hand_L, origin_x_hand_R, origin_y_hand_R,
          thumb_finger_x_hand_L, thumb_finger_y_hand_L, thumb_finger_x_hand_R, thumb_finger_y_hand_R,
    index_finger_x_hand_L, index_finger_y_hand_L, index_finger_x_hand_R, index_finger_y_hand_R,
    middle_finger_x_hand_L, middle_finger_y_hand_L, middle_finger_x_hand_R, middle_finger_y_hand_R,
    ring_finger_x_hand_L, ring_finger_y_hand_L, ring_finger_x_hand_R, ring_finger_y_hand_R,
    pinky_finger_x_hand_L, pinky_finger_y_hand_L, pinky_finger_x_hand_R, pinky_finger_y_hand_R)
  #for hand_world_landmarks in results.multi_hand_world_landmarks:
  #  mp_drawing.plot_landmarks(
  #    hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
  return (hand_found_R, hand_found_L, origin_x_hand_L, origin_y_hand_L, origin_x_hand_R, origin_y_hand_R,
          thumb_finger_x_hand_L, thumb_finger_y_hand_L, thumb_finger_x_hand_R, thumb_finger_y_hand_R,
  index_finger_x_hand_L, index_finger_y_hand_L, index_finger_x_hand_R, index_finger_y_hand_R,
  middle_finger_x_hand_L, middle_finger_y_hand_L, middle_finger_x_hand_R, middle_finger_y_hand_R,
  ring_finger_x_hand_L, ring_finger_y_hand_L, ring_finger_x_hand_R, ring_finger_y_hand_R,
  pinky_finger_x_hand_L, pinky_finger_y_hand_L, pinky_finger_x_hand_R, pinky_finger_y_hand_R)