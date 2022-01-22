import argparse
import cv2
import math
import time
import numpy as np
import glob
import os
import tqdm
from videoprocessor import VideoProcessorSK

parser = argparse.ArgumentParser()
parser.add_argument('--videos_path', type=str, default='./sample_videos', help='path to saved videos')
parser.add_argument('--export_images_path', type=str, default='/content/drive/MyDrive/ResearchProject/videos-bright-6', help='high level path to exported images')
parser.add_argument('--batch_num', type=int, default='123', help='batch number for subfolder of exported images')

args = parser.parse_args()


input_path = args.videos_path
images_folder = args.export_images_path + "/images-batch-" + str(args.batch_num) + "/"
videos = np.sort([fn for fn in glob.glob(input_path+'/*') if "Labeled" not in fn])
print('filenames:')
print(videos)

for ivid,vid in enumerate(videos):
    tic = time.time()
    print(vid)
    vidname = os.path.basename(vid)
    vname = vidname.split('.')[0]
    
    if os.path.isfile(os.path.join(input_path,vname + '_openposeLabeled.mp4')):
        print("Labeled video already created.")
    else:
        # break into frames
        clip = VideoProcessorSK(fname = os.path.join(input_path,vidname),sname = os.path.join(input_path,vname + '_openposeLabeled.mp4'))# input name, output name
        ny = clip.height()
        nx = clip.width()
        fps = clip.fps()
        nframes = clip.frame_count()
        duration = nframes/fps
        print("Duration of video [s]: ", duration, ", recorded with ", fps,
              "fps!")
        print("Overall # of frames: ", nframes, "with frame dimensions: ",
              ny,nx)
        print("Generating frames")
        frame_number = 1
        for index in tqdm(range(nframes)):
            input_image = clip.load_frame()
            second = math.floor(index/fps)
            try:
                real_frame_number = index
                filename = images_folder + vname + "_" + str(real_frame_number) + "_" + str(frame_number) + ".jpg"
                if (index/fps > second and index/fps <= second + 1/fps ):
                    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(filename, image)
                    print(filename)
                    frame_number +=1
                elif (index/fps > second + 1./3 and index/fps <= second + 1./3 + 1/fps):
                    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(filename, image)
                    print(filename)
                    frame_number += 1
                elif (index/fps > second + 1./3*2 and index/fps <= second + 1./3*2 + 1/fps):
                    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(filename, image)
                    print(filename)
                    frame_number += 1
            except Exception as e:
                print(repr(e))
                print('error during image extraction')
        clip.close()
        toc = time.time()
        print ('processing time is %.5f' % (toc - tic))
