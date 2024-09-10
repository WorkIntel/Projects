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
#from scipy.spatial.transform import Rotation as Rot
#from scipy import interpolate 
from scipy.interpolate import RegularGridInterpolator 

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from opencv_realsense_camera import RealSense

# import logging as log
# log.basicConfig(stream=sys.stdout, level=log.DEBUG, format='[%(asctime)s.%(msecs)03d] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',  datefmt="%M:%S")
# log.getLogger('matplotlib.font_manager').disabled = True
# log.getLogger('matplotlib').setLevel(log.WARNING)
# log.getLogger('PIL').setLevel(log.WARNING)

import matplotlib.pyplot as plt

#%% Logger
import logging
log         = logging.getLogger("robot")
#formatter   = logging.Formatter('[%(asctime)s.%(msecs)03d] {%(filename)6s:%(lineno)3d} %(levelname)s - %(message)s', datefmt="%M:%S", style="{")
formatter   = logging.Formatter('[%(asctime)s] - [%(filename)12s:%(lineno)3d] - %(levelname)s - %(message)s')
log.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# file_handler = logging.FileHandler("main_app.log", mode="a", encoding="utf-8")
# file_handler.setLevel("WARNING")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


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

        elif img_type == 4:
            self.imgD = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_029.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_031.png", cv.IMREAD_GRAYSCALE)
            self.imgR = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_030.png", cv.IMREAD_GRAYSCALE)              

        elif img_type == 12:
            self.imgD = cv.pyrDown(cv.imread(r"C:\Data\Corr\d2_Depth.png", cv.IMREAD_GRAYSCALE))
            self.imgL = cv.pyrDown(cv.imread(r"C:\Data\Corr\l2_Infrared.png", cv.IMREAD_GRAYSCALE))
            self.imgR = cv.pyrDown(cv.imread(r"C:\Data\Corr\r2_Infrared.png", cv.IMREAD_GRAYSCALE) )  

        elif img_type == 13:
            self.imgD = cv.pyrDown(cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE))
            self.imgL = cv.pyrDown(cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE))
            self.imgR = cv.pyrDown(cv.imread(r"C:\Data\Corr\r3_Infrared.png", cv.IMREAD_GRAYSCALE) )             

        elif img_type == 21:  # Test one image against image 
            image1      = np.random.randn(240, 320) * 60 + 60
            image2      = image1.copy()
            shift       = np.array([-5, -15])*1
            image1      = np.roll(image1, shift, axis=(0, 1))  
            self.imgL, self.imgR, self.imgD = np.uint8(image1), np.uint8(image2), np.uint8(image1)

        elif img_type == 22:
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)
            #self.imgR = cv.imread(r"C:\Data\Corr\r3_Infrared.png", cv.IMREAD_GRAYSCALE)    
            self.imgR = np.roll(self.imgL, np.array([-5, 15]), axis=(0, 1)) 

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
        self.imgD = cv.convertScaleAbs(frame[:,:,2], alpha=3) 
        #self.imgD = cv.normalize(frame[:,:,2], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
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
        flow            = cv.calcOpticalFlowFarneback(grayL, grayR, None, 0.5, 3, 5, 3, 5, 1.2, 0)

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
        return flow  

    def block_matching(self, f1, f2, block_size=8, search_range=16):
        """
        Performs block matching motion estimation.

        Args:
            f1: The anchor frame image.
            f2: The target frame image.
            block_size: The size of the blocks (default: 8).
            search_range: The maximum displacement for the search window (default: 16).

        Returns:
            A tuple (fp, mvx, mvy) where:
                - fp: The predicted image.
                - mvx: The horizontal motion vector field.
                - mvy: The vertical motion vector field.
        """

        height, width   = f1.shape[:2]
        fp              = np.zeros_like(f1)
        mvx             = np.zeros((height // block_size, width // block_size))
        mvy             = np.zeros((height // block_size, width // block_size))

        search_range_r  = search_range>>2
        search_range_c  = search_range

        for i in range(0, height - block_size + 1, block_size): # r
            for j in range(0, width - block_size + 1, block_size): # c
                mad_min = np.inf
                best_c, best_r = 0, 0
                block1 = f1[i:i + block_size, j:j + block_size]
                for r in range(-search_range_r, search_range_r + 1): # r
                    for c in range(-search_range_c, search_range_c + 1): # c
                        if i + r >= 0 and i + r + block_size - 1 < height and  j + c >= 0 and j + c + block_size - 1 < width:
                            
                            block2 = f2[i + r:i + r + block_size, j + c:j + c + block_size]
                            mad = np.sum(np.abs(block1 - block2))

                            if mad < mad_min:
                                mad_min = mad
                                best_r, best_c = r, c

                block3  = f2[i + best_r:i + best_r + block_size, j + best_c:j + best_c + block_size]
                fp[i:i + block_size, j:j + block_size] = block3
                iblk = i // block_size + 1
                jblk = j // block_size + 1
                mvx[iblk - 1, jblk - 1] = best_c
                mvy[iblk - 1, jblk - 1] = best_r

        # interpolate to the size of the original image
        x      = np.arange(0, mvx.shape[1]) * block_size
        y      = np.arange(0, mvx.shape[0]) * block_size
        #
        #data   = ff(xg, yg)
        xi      = np.arange(0, width)
        yi      = np.arange(0, height)  

        # 
        xg, yg = np.meshgrid(xi, yi, indexing='ij')
        test_points = np.array([xg.ravel(), yg.ravel()]).T

     
        interp = RegularGridInterpolator([y, x], mvx, bounds_error=False, fill_value=None)
        mvxi   = interp(test_points, method='linear').reshape((height,width))
        interp = RegularGridInterpolator([y, x], mvy, bounds_error=False, fill_value=None)
        mvyi   = interp(test_points, method='linear').reshape((height,width))

        flow    = np.stack((mvxi,mvyi),axis = 2)
        return fp, flow  
    
    def block_match_in_search_region(self, match_block, search_region):
        "single block matching in search region"
        #block1 = f1[i:i + block_size, j:j + block_size]
        block_size_r    = match_block.shape[0]
        block_size_c    = match_block.shape[1]
        search_range_r  = search_region.shape[0]
        search_range_c  = search_region.shape[1]
        best_c, best_r  = 0, 0
        
        search_result   = np.zeros((search_range_r - block_size_r,search_range_c - block_size_c))

        for r in range(0, search_range_r - block_size_r ): # r
            for c in range(0, search_range_c - block_size_c ): # c
                block2              = search_region[r:r + block_size_r,c:c + block_size_c]
                mad                 = np.sum(np.abs(match_block - block2))
                search_result[r,c]  = mad

        # find first minima
        best_r, best_c = np.unravel_index(search_result.argmin(), search_result.shape)

        # find secong minima
        suppres_size   = 5
        r_min, r_max   = np.maximum(0,best_r-suppres_size),np.minimum(search_range_r,best_r+suppres_size)
        c_min, c_max   = np.maximum(0,best_c-suppres_size),np.minimum(search_range_c,best_c+suppres_size)
        best_v         = search_result[r_min:r_max,c_min:c_max].min()
        search_result[r_min:r_max,c_min:c_max]= 1e9
        best_v2        = search_result.min()
        best_conf      = 1 - best_v/best_v2

        return  best_r, best_c, best_conf   

    def block_matching_with_confidence(self, f1, f2, block_size=8, search_range=16):
        """
        Performs block matching motion estimation also computes confidence.

        Args:
            f1: The anchor frame image.
            f2: The target frame image.
            block_size: The size of the blocks (default: 8).
            search_range: The maximum displacement for the search window (default: 16).

        Returns:
            A tuple (fp, mvx, mvy) where:
                - fp:  The predicted image.
                - mvx: The horizontal motion vector field.
                - mvy: The vertical motion vector field.
                - conf:The confidenc eof the peak 
        """

        height, width   = f1.shape[:2]
        fp              = np.zeros_like(f1)
        mvx             = np.zeros((height // block_size, width // block_size))
        mvy             = np.zeros((height // block_size, width // block_size))
        mvc             = np.zeros((height // block_size, width // block_size)) # confidence

        # up and down search
        block_size_r    = block_size>>1
        search_range_r  = np.maximum(search_range>>1, block_size_r)
        # only to the right
        search_range_c  = np.maximum(search_range, block_size)

        # rows can be negative and positive. columns only positive because of Left and Right stereo
        for i in range(search_range_r, height - search_range_r + 1, block_size): # r
            for j in range(0, width - search_range_c + 1, block_size): # c

                match_block     = f1[i-block_size_r  :i + block_size_r,  j:j + block_size]
                search_region   = f2[i-search_range_r:i + search_range_r,j:j + search_range_c]

                best_r, best_c, best_v = self.block_match_in_search_region(match_block, search_region) 
                block3          = search_region[best_r:best_r + block_size, best_c:best_c + block_size]
                best_r          = best_r - block_size_r # offset

                #block3          = f2[i + best_r:i + best_r + block_size, j + best_c:j + best_c + block_size]
                fp[i:i + block_size, j:j + block_size] = block3
                iblk            = i // block_size 
                jblk            = j // block_size 
                mvx[iblk, jblk] = best_c
                mvy[iblk, jblk] = best_r
                mvc[iblk, jblk] = best_v
                       

        # interpolate to the size of the original image
        x      = np.arange(0, mvx.shape[1]) * block_size
        y      = np.arange(0, mvx.shape[0]) * block_size
        #
        #data   = ff(xg, yg)
        xi      = np.arange(0, width)
        yi      = np.arange(0, height)  

        # 
        xg, yg = np.meshgrid(xi, yi, indexing='ij')
        test_points = np.array([xg.ravel(), yg.ravel()]).T

     
        interp = RegularGridInterpolator([y, x], mvx, bounds_error=False, fill_value=None)
        mvxi   = interp(test_points, method='linear').reshape((height,width))
        interp = RegularGridInterpolator([y, x], mvy, bounds_error=False, fill_value=None)
        mvyi   = interp(test_points, method='linear').reshape((height,width))
        interp = RegularGridInterpolator([y, x], mvc, bounds_error=False, fill_value=None)
        mvci   = interp(test_points, method='linear').reshape((height,width))
        flow    = np.stack((mvxi,mvyi,mvci),axis = 2)

        return fp, flow      

    # -----------------------------------------
    def show_images_left_right(self):
        "draw left right results"
        if self.imgL is None or self.imgR is None:
            self.tprint('No images found')
            return False
            
        # deal with black and white
        img_show = np.concatenate((self.imgL, self.imgR ), axis = 1)

        cv.imshow('Image L-R', img_show)
        #self.tprint('show done')
        ch = cv.waitKey(1)
        return True

    def show_images_depth(self):
        "draw results of depth estimation"
        if self.imgD is None and self.imgC is None:
            self.tprint('No images found')
            return False
        
        elif self.imgD is None:
            img_show = self.imgC

        elif np.all(self.imgD.shape == self.imgC.shape):
            img_show = np.concatenate((self.imgD, self.imgC ), axis = 1)

        else:
            self.imgD = np.repeat(self.imgD[:,:,np.newaxis], 3, axis = 2)
            #img_show = self.imgC #np.concatenate((self.imgD, self.imgC ), axis = 1)
            img_show = np.concatenate((self.imgD, self.imgC ), axis = 1)
            
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

    def show_flow(self, img, flow, step=16):
        "draw flow lines"
        QUIVER  = (255, 100, 0)
        h, w    = img.shape[:2]
        y, x    = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
        flowxy  = flow[:,:,:2] # x,y only
        fx, fy  = flowxy[y, x].T
        lines   = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines   = np.int32(lines + 0.5)
        if len(img.shape)<3:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        else:
            vis = img

        # check confidence
        if len(flow.shape) > 2:
            imgc = np.uint8(255*flow[:,:,2])
            cv.imshow('Flow Confidence', imgc)    
            ch = cv.waitKey(1)        

        cv.polylines(vis, lines, 0, QUIVER)
        for (x1, y1), (x2, y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
            cv.circle(vis, (x2, y2), 1, (0, 0, 255), -1)

        cv.imshow('Flow (q-exit)', vis)
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
        p.init_image(4)
        p.depth_opencv()
        p.show_images_depth()
        cv.waitKey()
        self.assertFalse(p.imgD is None)  
          
    def test_depth_opencv_advanced(self):
        "depth compute"
        p = DepthEstimator()
        p.init_image(4)
        p.depth_opencv_advanced()
        p.show_images_depth()
        cv.waitKey()
        self.assertFalse(p.imgD is None)  

    def test_video_stream_opencv_advanced(self):
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

    def test_show_flow(self):
        "draw flow"
        p = DepthEstimator()
        p.init_stream()
        ret  = False
        while not ret:
            p.read_stream()
            flow = p.dense_optical_flow()
            ret  = p.show_flow(p.imgL, flow)          

        self.assertFalse(p.imgD is None)  

    def test_block_matching(self):
        "depth from block matching - bruit force"
        p           = DepthEstimator()
        isOk        = p.init_image(3)
        isOk        = p.show_images_left_right()
        img, flow   = p.block_matching(p.imgL, p.imgR, block_size=16, search_range=32)
        isOk        = p.show_flow(img, flow, step=16)
        ch = cv.waitKey()
        self.assertFalse(p.imgD is None)          

    def test_block_matching_with_confidence(self):
        "depth from block matching - bruit force with confidence"
        p           = DepthEstimator()
        isOk        = p.init_image(22)
        isOk        = p.show_images_left_right()
        img, flow   = p.block_matching_with_confidence(p.imgL, p.imgR, block_size=16, search_range=32)
        isOk        = p.show_flow(img, flow, step=16)
        ch          = cv.waitKey()
        self.assertTrue(isOk)   



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
    #suite.addTest(TestDepthEstimator("test_video_stream_opencv_advanced")) # ok
    #suite.addTest(TestDepthEstimator("test_dense_optical_flow")) # so so
    #suite.addTest(TestDepthEstimator("test_show_flow"))
    #suite.addTest(TestDepthEstimator("test_block_matching"))
    suite.addTest(TestDepthEstimator("test_block_matching_with_confidence"))

    runner = unittest.TextTestRunner()
    runner.run(suite)

