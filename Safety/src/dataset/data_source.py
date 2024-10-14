#!/usr/bin/env python

'''

Data Source - brings data in different format for Laser Dtector

==================


Usage:

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 
    See README.md


'''

import numpy as np
import cv2 as cv
import unittest

from dataset.extract_images_from_ros1bag import read_bin_file

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from opencv_realsense_camera import RealSense
from common import log



# see update function
ESTIMATOR_OPTIONS = {1:'std',2:'std integrated',3:'contrast',4:'contrast',5:'contrast maxim',
                     11:'saturate',12:'texture', 21:'laser on-off', 
                     31:'iir', 
                     41:'dft-filter', 42:'spatial-filter',
                     51:'rosbag'}

        
#import logging as log
#log.basicConfig(stream=sys.stdout, level=log.DEBUG, format='[%(asctime)s.%(msecs)03d] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',  datefmt="%M:%S")
# log.getLogger('matplotlib.font_manager').disabled = True
# log.getLogger('matplotlib').setLevel(log.WARNING)
# log.getLogger('PIL').setLevel(log.WARNING)
# import matplotlib.pyplot as plt

# --------------------------------
#%% Helpers
def draw_str(dst, target, s):
    x, y = target
    dst = cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    dst = cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    return dst

# --------------------------------
#%% Data source
class DataSource:

    def __init__(self):

        # params
        self.mode            = 'ii2'  # video recording mode 
        self.vtype           = 0      # video source type
        self.frame_size      = (480,640)
        self.roi             = [0,0,self.frame_size[1],self.frame_size[0]]

        self.video_src       = None   # video source
        self.video_src2      = None   # video source 2 for two ros bags

        self.first_time      = True   # help to deal with initialization
        self.frame_count     = 0      # count frames
        
        self.frame_color     = None   # multiple channels - colors
        self.frame_gray      = None   # process separate channels
        self.frame_depth     = None   # process separate channels
        self.frame_left      = None   # raw video left frame
        self.frame_right     = None   # raw video right frame or previous left
        self.frame_show      = None   # show video frame
        self.figure_name     = 'Input'# figure name to show
        #self.frame_left2     = None   # left frame with previous pattern On-Off tests

        self.tprint('Source is defined')

    def init_video(self, video_type = 11):
        # create video for test
        #w,h                 = self.frame_size
        if video_type == 1:
            "RealSense"
            fmode           = 'iig' 
            self.video_src  = RealSense(mode=fmode)                   

        elif video_type == 11:
            "with pattern"
            fname           = r"C:\Users\udubin\Documents\Projects\Safety\data\laser_power\video_ii2_000.mp4"
            fmode           = 'ii2' 
            self.video_src  = cv.VideoCapture(fname)

        elif video_type == 21:
            "pattern on - off"
            fname           = r"C:\Users\udubin\Documents\Projects\Safety\data\laser_power\video_ii2_001_ponoff.mp4"
            fmode           = 'ii2'  
            self.video_src  = cv.VideoCapture(fname)

        elif video_type == 31:
            "pattern on - off from bag files"
            fname           = r"C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data\image_1726733151866586685_1280x720_step_1280_8uc1.bin"           
            self.video_src  = fname
            fname           = r"C:\Data\Safety\AGV\12_static_no_prj_covered_hall_carpet\12_static_no_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data\image_1726732880805986404_1280x720_step_1280_8uc1.bin"
            self.video_src2 = fname
            fmode           = 'img'  

        elif video_type == 32:
            "pattern on - off from bag files"
            fname           = r"C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data\image_1726733188232460976_1280x720_step_1280_8uc1.bin"           
            self.video_src  = fname
            fname           = r"C:\Data\Safety\AGV\12_static_no_prj_covered_hall_carpet\12_static_no_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data\image_1726732911605116129_1280x720_step_1280_8uc1.bin"
            self.video_src2 = fname
            fmode           = 'img' 

        else:
            self.tprint(f'Video type {video_type} is not supported','E')
            raise ValueError

        # read one frame to init
        self.mode           = fmode  
        self.vtype          = video_type
        #ret                 = self.init_data()
        self.tprint('Work mode : %s, video type : %d' %(fmode,video_type))
        return True
      
    def init_roi(self, roi_type = 1):
        "specify the relevant region"
        h,w     = self.frame_size
        roi     = [0,0,w,h]
        h2,w2   = h >>1, w>>1
        if roi_type == 1:
            roi = [w2-100,h2-100,w2+100,h2+100] # xlu, ylu, xrb, yrb
        elif roi_type == 2:
            roi = [w2-200,h2-200,w2+200,h2+200] # xlu, ylu, xrb, yrb
        else:
            pass
            
        self.tprint('ROI position %s' %str(roi))

        #self.roi = roi       
        return roi 

    def convert_frame_from_input(self, frame):
        "extract depth and gray level channels"

        if self.mode == 'rgb':
            self.frame_left  = frame[:,:,0] 
            self.frame_right = frame[:,:,1]
            self.frame_gray  = frame[:,:,2]  
        elif self.mode == 'iig':
            self.frame_left  = frame[:,:,0] 
            self.frame_right = frame[:,:,1]
            self.frame_gray  = frame[:,:,2]   
        elif self.mode == 'ii2':
            self.frame_left  = frame[:,:,0] 
            self.frame_right = frame[:,:,1]
            self.frame_gray  = frame[:,:,2]                              
        else:
            self.tprint('bad mode','E')
            raise ValueError('bad mode')

        return True
    
    def get_image(self):
        "get a 2 infrared image frame from 2 files"
        
        fsize               = (1280,720)
        fbpp                = 8

        fname               = self.video_src 
        img_array           = read_bin_file(fname,fsize,fbpp)
        self.frame_left     = img_array 

        fname               = self.video_src2 
        img_array           = read_bin_file(fname,fsize,fbpp)
        self.frame_right    = img_array     

        self.frame_gray     = self.frame_right - self.frame_left    
        self.frame_color    = np.stack((self.frame_left , self.frame_right, self.frame_left  ), axis = 2)
 
        frame_out            = self.frame_color # stam
        self.first_time      = True
        self.frame_count     = self.frame_count + 1
        return True, frame_out.astype(np.float32)      

    def get_frame(self):
        "get a single frame from the stream"
        # as the video frame.
        ret, frame              = self.video_src.read()  
        if not ret:
            return ret, []
                
        # convert channels
        ret                     = self.convert_frame_from_input(frame)
 
        frame_out            = frame
        self.first_time      = False
        self.frame_count     = self.frame_count + 1
        return True, frame_out.astype(np.float32)   
    
    def get_data(self):
        "get all the data structures"
        #self.first_time     = True
        #self.frame_count    = 0
        #self.integration_enb = True   # enable integration process

        if self.mode == 'img':
            # read image data
            ret, frame          = self.get_image()

        else: 
            # read video data
            ret, frame          = self.get_frame()  

        if not ret:
            self.tprint(f'Data source is not found ','E')
            return ret

        return ret    

    def show_data(self):
        "draw relevant image data"
        if self.frame_left is None or self.frame_right is None:
            self.tprint('No images found')
            return False
            
        # deal with black and white
        #img_show    = np.concatenate((self.frame_left, self.frame_right), axis = 1)
        img_show    = self.frame_color

        while img_show.shape[1] > 2000:
            img_show    = cv.resize(img_show, (img_show.shape[1]>>1,img_show.shape[0]>>1), interpolation=cv.INTER_LINEAR)

        cv.imshow('Image L-R', img_show)
        #self.tprint('show done')
        ch  = cv.waitKey(1)
        ret = ch != ord('q')
        return ret

    def finish(self):
        # Close down the video stream
        if self.vtype < 10:
            self.video_src.release()

    def tprint(self, txt = '', level = 'I'):
        if level == "I":
            log.info(txt)
        elif level == "W":
            log.warning(txt)
        elif level == "E":
            log.error(txt)
        else:
            log.info(txt)     

# --------------------------------
#%% Tests
class TestDataSource(unittest.TestCase):

    def test_data_source_rs(self):
        "show image from camera"
        p       = DataSource()
        srcid   = 1
        ret     = p.init_video(srcid)
        while ret:
            ret     = p.get_data()
            ret     = p.show_data()
        p.finish()
        self.assertFalse(ret)
  
    def test_data_source_video(self):
        "show image from video file"
        p       = DataSource()
        srcid   = 11
        ret     = p.init_video(srcid)
        while ret:
            ret     = p.get_data()
            ret     = p.show_data() and ret
        p.finish()
        self.assertFalse(ret)

    def test_rosbag_data_show(self):
        "image data with on off of two frames"
        srcid   = 31       # video tytpe

        d       = DataSource()
        ret     = d.init_video(srcid)
        ret     = d.get_data()
        ret     = d.show_data() 
        cv.waitKey()
        self.assertTrue(not ret)      

    def test_rosbag_data_with_pattern_switch(self):
        "image data with on off of two frames"
        srcid   = 32       # ros bag 31,32

        d       = DataSource()
        ret     = d.init_video(srcid)
        ret     = d.get_data()
        reti    = d.show_data()
        cv.waitKey()
        self.assertTrue(not ret)         

# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestDataSource("test_data_source_rs")) # ok
    #suite.addTest(TestDataSource("test_data_source_video")) # ok
    #suite.addTest(TestDataSource("test_rosbag_data_show")) # ok
    suite.addTest(TestDataSource("test_rosbag_data_with_pattern_switch")) # ok
    
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    #print(__doc__)

    RunTest()
    #RunApp('iig').run()    

