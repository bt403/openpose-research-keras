from skimage.feature import hog
from PIL import Image, ImageOps
import csv as csv
import pandas as pd
import argparse

hog_features = []
hog_features_flipped = []

hog_features_eyes = []
hog_features_eyes_flipped = []

hog_features_mouth = []
hog_features_mouth_flipped = []
ppc = 16

parser = argparse.ArgumentParser()
parser.add_argument('--processed_features', type=str, default='./data/estimates/processed_features.pkl')
parser.add_argument('--output_path', type=str, default='./data/estimates/')
parser.add_argument('--eyes_folder', type=str, default="/content/drive/MyDrive/ResearchProject/eyes/")
parser.add_argument('--mouth_folder', type=str, default="/content/drive/MyDrive/ResearchProject/mouth/")

args = parser.parse_args()

output_path = args.output_path

def getHogFeatures(path, size=(128,128)):
  image_org = Image.open(path)
  image_org = image_org.resize(size)
  image = ImageOps.grayscale(image_org)
  image_flipped = ImageOps.mirror(image)
  fd = hog(image, orientations=9, pixels_per_cell=(ppc,ppc),cells_per_block=(2,2), block_norm="L2-Hys")
  fd_flipped = hog(image_flipped, orientations=9, pixels_per_cell=(ppc,ppc),cells_per_block=(2,2), block_norm="L2-Hys")
  return fd, fd_flipped, image_org

pdata = pd.read_pickle(args.processed_features)

for i, row in pdata.iterrows():
  print(i)
  path = row['path_cropped']
  print(path)
  img_name = "_".join(path.split("/")[-1].split("_")[1:-1]) + ".jpg"
  path_eyes = args.eyes_folder + img_name
  path_mouth = args.mouth_folder + img_name

  fd, fd_flipped, image_1 = getHogFeatures(path)
  fd_eyes, fd_eyes_flipped, image_2 = getHogFeatures(path_eyes, (128,64))
  fd_mouth, fd_mouth_flipped, image_3 = getHogFeatures(path_mouth, (128,64))

  hog_features.append(fd)
  hog_features_flipped.append(fd_flipped)

  hog_features_eyes.append(fd_eyes)
  hog_features_eyes_flipped.append(fd_eyes_flipped)

  hog_features_mouth.append(fd_mouth)
  hog_features_mouth_flipped.append(fd_mouth_flipped)

hog_features_1 = pd.DataFrame(hog_features)
hog_features_2 = pd.DataFrame(hog_features_flipped)
hog_features_3 = pd.DataFrame(hog_features_eyes)
hog_features_4 = pd.DataFrame(hog_features_eyes_flipped)
hog_features_5 = pd.DataFrame(hog_features_mouth)
hog_features_6 = pd.DataFrame(hog_features_mouth_flipped)

hog_features_1.to_pickle(output_path + "hog_features_face.pkl")
hog_features_2.to_pickle(output_path + "hog_features_face_flipped.pkl")
hog_features_3.to_pickle(output_path + "hog_features_eyes.pkl")
hog_features_4.to_pickle(output_path + "hog_features_eyes_flipped.pkl")
hog_features_5.to_pickle(output_path + "hog_features_mouth.pkl")
hog_features_6.to_pickle(output_path + "hog_features_mouth_flipped.pkl")