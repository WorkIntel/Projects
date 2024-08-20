#!/usr/bin/env python

'''
Compoare Depth Estimation - RealSense against Other Correlators


Usage:

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 



'''

import numpy as np
import cv2 as cv
import unittest
from scipy.spatial.transform import Rotation as Rot
from scipy import interpolate 

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from opencv_realsense_camera import RealSense

import logging as log
log.basicConfig(stream=sys.stdout, level=log.DEBUG, format='[%(asctime)s.%(msecs)03d] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',  datefmt="%M:%S")
log.getLogger('matplotlib.font_manager').disabled = True
log.getLogger('matplotlib').setLevel(log.WARNING)
log.getLogger('PIL').setLevel(log.WARNING)

import matplotlib.pyplot as plt


#%% Main
class DepthEstimator:
    def __init__(self):

        self.frame_size = (640,480)
        self.imgD        = None  # depth from real sense
        self.imgC        = None  # estimation results
        self.imgL        = None
        self.imgR        = None

        # params
        self.MIN_STD_ERROR   = 0.01

        # stream
        self.cap         = None


    def init_image(self, img_type = 1):
        # create some images for test
        w,h             = self.frame_size        

        if img_type == 1:
            self.imgD = cv.imread(r"C:\Data\Corr\d1_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\r1_Infrared.png", cv.IMREAD_GRAYSCALE)
            self.imgR = cv.imread(r"C:\Data\Corr\l1_Infrared.png", cv.IMREAD_GRAYSCALE)   

        elif img_type == 2:
            self.imgD = cv.imread(r"C:\Data\Corr\d2_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l2_Infrared.png", cv.IMREAD_GRAYSCALE)
            self.imgR = cv.imread(r"C:\Data\Corr\r2_Infrared.png", cv.IMREAD_GRAYSCALE)   
            
        elif img_type == 3:
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)
            self.imgR = cv.imread(r"C:\Data\Corr\r3_Infrared.png", cv.IMREAD_GRAYSCALE)    
        else:
            self.tprint('Incorrect image type to load')        

                                    
        self.imgC = self.imgD    
        #self.img        = np.uint8(self.img) 

        #self.img = self.add_noise(self.img, 0)
        #self.img = cv.resize(self.img , dsize = self.frame_size)   
        #imgL = cv.pyrDown(self.imgL)  # downscale images for faster processing
        #imgR = cv.pyrDown(self.imgR)
              
        return True
    
    def init_stream(self):
        "read data from real sense"
        self.cap = RealSense(mode = 'iid', use_ir = True)

    def read_stream(self):
        "reading data stream from RS"
        if self.cap is None:
            self.tprint('init stream first')
            return
        
        # frame is I1,I2, D data
        ret, frame = self.cap.read()
        if ret is False:
            return
        
        # assign
        self.imgL = frame[:,:,0]
        self.imgR = frame[:,:,1]
        self.imgD = cv.convertScaleAbs(frame[:,:,2], alpha=10) 
        return
      
    def init_roi(self, test_type = 1):
        "load the test case"
        roi = [0,0,self.frame_size[0],self.frame_size[1]]
        if test_type == 1:
            roi = [310,230,330,250] # xlu, ylu, xrb, yrb
        elif test_type == 2:
            roi = [300,220,340,260] # xlu, ylu, xrb, yrb
        elif test_type == 3:
            roi = [280,200,360,280] # xlu, ylu, xrb, yrb            
        elif test_type == 4:
            roi = [220,140,420,340] # xlu, ylu, xrb, yrb      
        elif test_type == 4:
            roi = [200,120,440,360] # xlu, ylu, xrb, yrb            
        return roi 

    def convert_disparity_to_depth(self):
        "from GIL"
        focal_len           = 175.910019
        baseline            = 94.773
        #replacementDepth    = focal_len *  baseline / (RectScaledInfra1.x - (maxLoc.x + RectScaledInfra2.x)); 

     
    def depth_opencv(self):
        "computes depth from L and R images"
        self.tprint('start processing')
        # Load the left and right images in grayscale
        left_image  = self.imgL
        right_image = self.imgR

        # Initialize the stereo block matching object
        stereo = cv.StereoBM_create(numDisparities=128, blockSize=15)

        # Compute the disparity map
        disparity = stereo.compute(left_image, right_image)

        # Normalize the disparity for display
        disparity_normalized = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        # Display the disparity map
        self.imgC =  disparity_normalized
        self.tprint('stop processing')
        return True


    def depth_opencv_advanced(self):
        "image computation using more elaborate features"   
        self.tprint('start processing')


        # disparity range is tuned for 'aloe' image pair
        window_size = 3
        min_disp = 16
        num_disp = 128 #112-min_disp
        stereo = cv.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 16,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )

        
        disparity            = stereo.compute(self.imgL, self.imgR) #.astype(np.float32) / 16.0

        disparity_normalized = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        self.imgC            = disparity_normalized
        self.tprint('Computing disparity... Done')
        return True
    
    def dense_optical_flow(self):
        "image computation using more elaborate features"   
        self.tprint('start processing')

        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        grayL           = self.imgL
        grayR           = self.imgR
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow            = cv.calcOpticalFlowFarneback(grayL, grayR, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        # Creates an image filled with zero intensities with the same dimensions as the frame
        mask            = np.zeros((grayL.shape[0],grayL.shape[1],3), dtype = grayL.dtype)
        # Sets image saturation to maximum
        mask[..., 1]    = 255        
        mask[..., 0]    = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2]    = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb                 = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        self.imgC           = rgb
        self.tprint('Computing disparity... Done')
        return True    

    # -----------------------------------------
    def show_images_left_right(self):
        "draw left right results"
        if self.imgL is None or self.imgR is None:
            self.tprint('No images found')
            
        # deal with black and white
        img_show = np.concatenate((self.imgL, self.imgR ), axis = 1)

        cv.imshow('Image L-R', img_show)
        #self.tprint('show done')
        ch = cv.waitKey()

    def show_images_depth(self):
        "draw results of depth estimation"
        if self.imgD is None and self.imgC is None:
            self.tprint('No images found')
            return False
        
        elif self.imgD is None:
            img_show = self.imgC
        else:
            #self.imgD = np.repeat(self.imgD, 3, axis = 2)
            img_show = self.imgC #np.concatenate((self.imgD, self.imgC ), axis = 1)
            
        # deal with black and white
        
        if img_show.shape[1] > 1600:
            img_show = cv.pyrDown(img_show)

        # # deal with black and white
        # img_show = np.uint8(img_show) #.copy()
        # if len(img_show.shape) < 3:
        #     img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)        

        cv.imshow('Image D-C (q-exit)', img_show)
        #self.tprint('show done')
        ch = cv.waitKey(1)
        ret = ch == ord('q')
        return ret



    def tprint(self, txt = '', level = 'I'):
        if level == "I":
            log.info(txt)
        elif level == "W":
            log.warning(txt)
        elif level == "E":
            log.error(txt)
        else:
            log.info(txt)
  


# ----------------------
#%% Tests
class TestDepthEstimator(unittest.TestCase):

    def test_show_images_left_right(self):
        "left right test"
        p = DepthEstimator()
        p.init_image(1)
        p.show_images_left_right()
        self.assertFalse(p.imgD is None)

    def test_show_images_depth(self):
        "depth show"
        p = DepthEstimator()
        p.init_image(1)
        p.show_images_depth()
        self.assertFalse(p.imgD is None)   

    def test_depth_opencv(self):
        "depth compute"
        p = DepthEstimator()
        p.init_image(3)
        p.depth_opencv()
        p.show_images_depth()
        self.assertFalse(p.imgD is None)  
          
    def test_depth_opencv_advanced(self):
        "depth compute"
        p = DepthEstimator()
        p.init_image(2)
        p.depth_opencv_advanced()
        p.show_images_depth()
        self.assertFalse(p.imgD is None)  

    def test_read_stream(self):
        "depth compute"
        p = DepthEstimator()
        p.init_stream()
        ret  = False
        while not ret:
            p.read_stream()
            p.depth_opencv_advanced()
            ret = p.show_images_depth()

        self.assertFalse(p.imgD is None) 

    def test_dense_optical_flow(self):
        "depth compute using optical flow"
        p = DepthEstimator()
        p.init_stream()
        ret  = False
        while not ret:
            p.read_stream()
            p.dense_optical_flow()
            ret = p.show_images_depth()

        self.assertFalse(p.imgD is None) 

# ----------------------
#%% App
class App:
    def __init__(self, src):
        self.cap   = RealSense()
        self.cap.change_mode('dep')

        self.frame = None
        self.paused = False
        self.tracker = DepthEstimator()

        cv.namedWindow('plane')

    def run(self):
        while True:
            playing = not self.paused
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            vis = self.frame.copy()
            if playing:
                tracked = self.tracker.track(self.frame)
                for tr in tracked:
                    cv.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                    for (x, y) in np.int32(tr.p1):
                        cv.circle(vis, (x, y), 2, (255, 255, 255))

            self.rect_sel.draw(vis)
            cv.imshow('plane', vis)
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break

if __name__ == '__main__':
    #print(__doc__)

    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestDepthEstimator("test_show_images_left_right"))
    #suite.addTest(TestDepthEstimator("test_show_images_depth"))
    #suite.addTest(TestDepthEstimator("test_depth_opencv"))
    #suite.addTest(TestDepthEstimator("test_depth_opencv_advanced"))
    #suite.addTest(TestDepthEstimator("test_read_stream"))
    suite.addTest(TestDepthEstimator("test_dense_optical_flow"))


    runner = unittest.TextTestRunner()
    runner.run(suite)

