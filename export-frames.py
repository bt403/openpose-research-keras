import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import pickle
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import skvideo.io
import skvideo
import glob
import os
from tqdm import tqdm
import pandas as pd

class VideoProcessor(object):
    '''
    Base class for a video processing unit, 
    implementation is required for video loading and saving
    '''
    def __init__(self,fname='',sname='', nframes = -1, fps = 30):
        self.fname = fname
        self.sname = sname

        self.nframes = nframes
        
        self.h = 0 
        self.w = 0
        self.sh = 0
        self.sw = 0
        self.FPS = fps
        self.nc = 3
        self.i = 0
        
        try:
            if self.fname != '':
                self.vid = self.get_video()
                self.get_info()
            if self.sname != '':
                self.sh = self.h
                self.sw = self.w
                self.svid = self.create_video()

        except Exception as ex:
            print('Error: %s', ex)
            
    def load_frame(self):
        try:
            frame = self._read_frame()
            self.i += 1
            return frame
        except Exception as ex:
            print('Error: %s', ex)
    
    def height(self):
        return self.h
    
    def width(self):
        return self.w
    
    def fps(self):
        return self.FPS
    
    def counter(self):
        return self.i
    
    def frame_count(self):
        return self.nframes
        
                       
    def get_video(self):
        '''
        implement your own
        '''
        pass
    
    def get_info(self):
        '''
        implement your own
        '''
        pass

    def create_video(self):
        '''
        implement your own
        '''
        pass
    

        
    def _read_frame(self):
        '''
        implement your own
        '''
        pass
    
    def save_frame(self,frame):
        '''
        implement your own
        '''
        pass
    
    def close(self):
        '''
        implement your own
        '''
        pass


class VideoProcessorSK(VideoProcessor):
    '''
    Video Processor using skvideo.io
    requires sk-video in python,
    and ffmpeg installed in the operating system
    '''
    def __init__(self, *args, **kwargs):
        super(VideoProcessorSK, self).__init__(*args, **kwargs)
    
    def get_video(self):
         return skvideo.io.FFmpegReader(self.fname)
        
    def get_info(self):
        infos = skvideo.io.ffprobe(self.fname)['video']
        self.h = int(infos['@height'])
        self.w = int(infos['@width'])
        self.FPS = eval(infos['@avg_frame_rate'])
        vshape = self.vid.getShape()
        all_frames = vshape[0]
        self.nc = vshape[3]

        if self.nframes == -1 or self.nframes>all_frames:
            self.nframes = all_frames
            
    def create_video(self):
        return skvideo.io.FFmpegWriter(self.sname, outputdict={'-r':str(self.FPS)})

    def _read_frame(self):
        return self.vid._readFrame()
    
    def save_frame(self,frame):
        self.svid.writeFrame(frame)
    
    def close(self):
        #self.svid.close()
        self.vid.close()


    
input_path = './video_data'
keras_weights_file='./model/keras/model.h5'

videos = np.sort([fn for fn in glob.glob(input_path+'/*') if "Labeled" not in fn])
print('filenames:')
print(videos)

# os.chdir(input_path)
for ivid,vid in enumerate(videos):
    tic = time.time()
    df = pd.DataFrame()
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
                filename = "/content/drive/MyDrive/ResearchProject/videos-bright-6/images-batch-1/" + vidname + "_" + str(real_frame_number) + "_" + str(frame_number) + ".jpg"
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
                print('error during pose estimation')
        clip.close()
        toc = time.time()
        print ('processing time is %.5f' % (toc - tic))
    
os.chdir('../')

