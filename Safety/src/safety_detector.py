#!/usr/bin/env python

'''
Multi modal background estimator and intrusion detector

==================

Using depth and RGB image to estimate the background during the training phase.
After that it compares each new image with the estimated background


Usage:

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 



'''

import numpy as np
import cv2 as cv
import unittest

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

#%% Helpers

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    #cv.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
    return img


#%% Main
class background_estimator:
    def __init__(self):

        # params
        self.MIN_OBJECT_SIZE = 32       # minimal size of the object
        self.MIN_STD_ERROR   = 10       # noise per pixel
        self.MIN_VALID_DEPTH = 10       # pixel value below which the depth is not valid
        self.BACKGROUND_NUM  = 3        # number of different background to estimate


        self.frame_size      = (480,640)
        self.roi             = [0,0,self.frame_size[1],self.frame_size[0]]

        self.video_src       = None   # video source
        self.mode            = 'gdd'  # video recording mode 
        self.first_time      = True   # help to deal with initialization
        self.frame_count     = 0      # count frames
        
        #self.channel_num     = 1      # how many channels are processed - depends on the mode  
        self.frame_gray      = None   # process separate channels
        self.frame_depth     = None   # process separate channels
        self.frame_in        = None   # raw video frame
        self.frame_show      = None   # show video frame
        self.figure_name     = 'Input'# figure name to show

        # integration over time
        self.frame_gray_int  = None   # process separate channels
        self.frame_depth_int = None   # process separate channels
        self.integration_enb = True   # enable integration process
        self.show_data_int   = False  # swicth that controls which dta to show

    def init_data(self):
        "init all the data strcutures"
        self.first_time     = True
        self.frame_count    = 0
        self.integration_enb = True   # enable integration process

        ret, frame          = self.get_frame()  
        if not ret:
            self.tprint(f'Video source is not found ','E')

        self.frame_gray_int  = self.frame_gray   # process separate channels
        self.frame_depth_int = self.frame_depth   # process separate channels

        self.tprint('Detector is initialized')
        return ret

    def init_video(self, video_type = 11):
        # create video for test
        #w,h                 = self.frame_size
        if video_type == 1:
            "RealSense"
            fmode           = 'rgb' 
            self.video_src  = RealSense(mode=fmode)

        elif video_type == 2:
            "RealSense"
            fmode           = 'rgd' 
            self.video_src  = RealSense(mode=fmode)      

        elif video_type == 3:
            "RealSense"
            fmode           = 'ddd' 
            self.video_src  = RealSense(mode=fmode)                     

        elif video_type == 11:
            "background"
            fname           = r"..\\data\\video_gdd_backg.mp4"
            fname           = r"C:\Users\udubin\Documents\Projects\Safety\data\video_gdd_backg.mp4"
            fmode           = 'gdd' 
            self.video_src  = cv.VideoCapture(fname)

        elif video_type == 12:
            "intrusion"
            fname           = r"..\data\video_gdd_intrusion.mp4"
            fmode           = 'gdd'  
            self.video_src  = cv.VideoCapture(fname)
        else:
            self.tprint(f'Video type {video_type} is not supported','E')
            raise ValueError

        # read one frame to init
        self.mode           = fmode  
        ret                 = self.init_data()
        self.tprint('Work mode : %s' %fmode)
        return ret
      
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
            
        self.tprint('ROI size %s' %str(roi))

        #self.roi = roi       
        return roi  
    
    def convert_frame_from_input(self, frame):
        "extract depth and gray level channels"

        if self.mode == 'rgb':
            frame_out  = frame #cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_depth= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif self.mode == 'ddd':
            frame_out  = frame[:,:,0]
            frame_gray = frame[:,:,0]*0+128
            frame_depth= frame[:,:,1]
        elif self.mode == 'gdd':
            frame_out  = frame[:,:,:2]
            frame_gray = frame[:,:,0]
            frame_depth= frame[:,:,1]    
        elif self.mode == 'rgd':
            frame_out  = frame[:,:,:] 
            frame_gray = frame[:,:,0]
            frame_depth= frame[:,:,2]       
        else:
            self.tprint('bad mode','E')

        return frame_out, frame_gray, frame_depth

    def get_frame(self, roi_type = 0):
        "get a single frame from the stream"
        # as the video frame.
        ret, frame  = self.video_src.read()  
        if not ret:
            return ret, []
        
        # first time - inint some stuff
        if self.first_time:
            self.frame_size     = frame.shape[:2]
            self.roi            = self.init_roi(roi_type)
        
        # reduce image size
        x0,y0,x1,y1             = self.roi
        frame                   = frame[y0:y1,x0:x1,:]
                
        # convert channels
        frame_out, frame_gray, frame_depth = self.convert_frame_from_input(frame)

        # # number of channels
        # if self.first_time:    
        #     self.channel_num = 1 if len(frame_out.shape) < 3 else frame_out.shape[2]

        self.frame_gray      = frame_gray.astype(np.float32)   # process separate channels
        self.frame_depth     = frame_depth.astype(np.float32)   # process separate channels  
        self.frame_in        = frame
        self.first_time      = False
        self.frame_count     = self.frame_count + 1
        return True, frame_out.astype(np.float32)   

    def integrate_over_time(self):
        "uses valid pixels to integrate over the time depth and gray channels"
        
        alpha                         = 0.1 if self.integration_enb else 0.0

        # depth
        valid_b                        = self.frame_depth > self.MIN_VALID_DEPTH
        self.frame_depth_int[valid_b] += alpha*(self.frame_depth[valid_b] - self.frame_depth_int[valid_b])
        # gray
        self.frame_gray_int           += alpha*(self.frame_gray  - self.frame_gray_int)

        return True


    def detect_space_time(self):
        "find regions that differ from the background"
        depth_bool                  = np.abs(self.frame_depth - self.frame_depth_int) > self.MIN_STD_ERROR
        gray_bool                   = np.abs(self.frame_gray - self.frame_gray_int) > self.MIN_STD_ERROR

        # in some modes ignore gray
        if self.mode == 'ddd':
            detect_bool             = depth_bool
        else:
            detect_bool             = np.logical_and(depth_bool,gray_bool)

        # detect contours
        contours                    = cv.findContours(detect_bool.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours                    = contours[0] if len(contours) == 2 else contours[1]
        rect                        = []
        for cntr in contours:
            area                = cv.contourArea(cntr)
            #convex_hull         = cv.convexHull(cntr)
            #convex_hull_area    = cv.contourArea(convex_hull)
            #ratio               = area / convex_hull_area
            if area > self.MIN_OBJECT_SIZE:
                x,y,w,h             = cv.boundingRect(cntr)
                rect.append([x,y,x+w,y+h])

        return rect

        
    def convert_frame_for_show(self, frame = None):
        "converts different frame types to the uint8 3 colors"

        if self.show_data_int:
            frame_g, frame_d = self.frame_gray_int, self.frame_depth_int
        else:
            frame_g, frame_d = self.frame_gray, self.frame_depth


        if self.mode == 'rgb':
            img_show    = np.uint8(frame_g) #.copy()
            img_show    = cv.applyColorMap(img_show, cv.COLORMAP_JET)
        elif self.mode == 'ddd':
            img_show    = np.uint8(frame_d)
            img_show    = cv.applyColorMap(img_show, cv.COLORMAP_JET)
        elif self.mode == 'gdd':
            img_show    = np.stack((frame_g,frame_d,frame_d),axis = 2)
            img_show    = np.uint8(img_show)
        elif self.mode == 'rgd':
            img_show    = np.uint8(frame_g) #.copy()
            #img_show    = cv.applyColorMap(img_show, cv.COLORMAP_JET)
            #img_show[:,:,2] = np.uint8(self.frame_depth) 
            img_show    = np.stack((img_show,img_show,np.uint8(frame_d)),axis = 2)
        else:
            self.tprint('bad mode','E')        
                
        # img_show     = np.uint8(frame) #.copy()
        # if len(img_show.shape) < 3:
        #     img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)  
        # else:
        #     if img_show.shape[2] == 2:
        #         img_show = np.stack((img_show[:,:,0],img_show[:,:,1],img_show[:,:,1]),axis = 2)

        self.frame_show  = img_show # keep it for click events
        return img_show 

    def show_click_event(self, event, x, y, flags, params): 
        # function to display the coordinates of the points clicked on the image  
        font        = cv.FONT_HERSHEY_SIMPLEX 
        font_size   = 0.6
        img         = self.frame_show
        b,g,r       = img[y, x, 0] , img[y, x, 1] , img[y, x, 2]       
        txt_to_show = f'[{x},{y}] = ({b},{g},{r})'
        
        # checking for left mouse clicks 
        if event == cv.EVENT_LBUTTONDOWN: 
    
            # displaying the coordinates on the image window             
            cv.putText(img, str(x) + ',' + str(y), (x,y), font,  font_size, (255, 0, 0), 2) 
            self.tprint(txt_to_show)  # displaying the coordinates on the Shell 
            cv.imshow(self.figure_name, img)   
    
        # checking for right mouse clicks      
        if event==cv.EVENT_RBUTTONDOWN: 
    
            # displaying the coordinates 
            cv.putText(img, txt_to_show,  (x,y), font, font_size,  (255, 255, 0), 2) 
            cv.drawMarker(img, (x,y),color=[0, 0, 255], thickness=3,  markerType= cv.MARKER_TILTED_CROSS, line_type=cv.LINE_AA,  markerSize=5)
            self.tprint(txt_to_show)  # displaying the coordinates on the Shell 
            cv.imshow(self.figure_name, img)       

    def show_image(self, img, figure_name = 'Input'):
        "draw results"
        if img is None:
            return
        self.figure_name    = figure_name

        # deal with frame shape
        img_show        = self.convert_frame_for_show(img)

        cv.imshow(figure_name, img_show)
        cv.setMouseCallback(figure_name, self.show_click_event) 
        #self.tprint('show done')
        ch = cv.waitKey()

    def show_video(self, figure_name = 'Input', roi_list = []):
        "draw results on video"
        # deal with black and white
        self.figure_name    = figure_name

        while True:
            ret, frame          = self.get_frame()
            if not ret:
                break

            # deal with frame shape
            img_show            = self.convert_frame_for_show(frame)

            # draw rois of processing
            for roi_p in roi_list:
                ix,iy,sx,sy = roi_p
                img_show    = cv.rectangle(img_show,(ix,iy),(sx,sy),(0,255,0),2)

            cv.imshow(figure_name, img_show)
            ch = cv.waitKey(30)
            if ch == ord('q'):
                break        

    def run_video_integration(self, figure_name = 'Input', roi_list = []):
        "show results on video with integration"
        # deal with black and white
        self.figure_name    = figure_name + ' : (i-int, t-train, r-reset, q- exit)'

        while True:
            ret, frame          = self.get_frame()
            if not ret:
                break

            # integrate over time
            ret                 = self.integrate_over_time()

            # find regions
            roi_list            = self.detect_space_time()

            # deal with frame shape
            img_show            = self.convert_frame_for_show(frame)

            # draw rois of processing
            for roi_p in roi_list:
                ix,iy,sx,sy = roi_p
                img_show    = cv.rectangle(img_show,(ix,iy),(sx,sy),(0,255,0),2)

            cv.imshow(self.figure_name, img_show)
            ch = cv.waitKey(30)
            if ch == ord('q'):
                break   
            elif ch == ord('i'):
                self.show_data_int = not self.show_data_int
                self.tprint('Switching show %s' %str(self.show_data_int))
            elif ch == ord('r'):
                self.init_data()  
            elif ch == ord('t'):
                self.integration_enb = not self.integration_enb
                self.tprint('Integration enb %s' %str(self.integration_enb))

    def finish(self):
        # Close down the video stream
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
class TestBackgroundEstimator(unittest.TestCase):

    def test_ImageShow(self):
        "show image and explore it by using click"
        p       = background_estimator()
        srcid   = 11
        ret     = p.init_video(srcid)
        p.show_image(p.frame_in,str(srcid))
        p.finish()
        self.assertFalse(p.frame_in is None)
  
    def test_VideoShow(self):
        "show video stream"
        p       = background_estimator()
        srcid   = 2
        ret     = p.init_video(srcid)
        p.show_video(str(srcid))
        p.finish()
        self.assertTrue(ret)  

    def test_VideoIntegration(self):
        "show video and integrate over time"
        p       = background_estimator()
        srcid   = 2
        ret     = p.init_video(srcid)
        p.run_video_integration(str(srcid))
        p.finish()
        self.assertTrue(ret) 

    def test_TrainingDetection(self):
        "show video and integrate over time upto 5 sec and then stop integration and do detect"
        p       = background_estimator()
        srcid   = 1  # 2 - detect gray and depth, 3 - only depth
        ret     = p.init_video(srcid)
        p.run_video_integration(str(srcid))
        p.finish()
        self.assertTrue(ret)         


# ----------------------
#%% App
class App:
    def __init__(self, src):
        self.cap   = RealSense()
        self.cap.change_mode('dep')

        self.frame = None
        self.paused = False
        self.tracker = PlaneMatcher()

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
    #suite.addTest(TestBackgroundEstimator("test_ImageShow")) # ok
    #suite.addTest(TestBackgroundEstimator("test_VideoShow")) # ok
    #suite.addTest(TestBackgroundEstimator("test_VideoIntegration")) # ok
    suite.addTest(TestBackgroundEstimator("test_TrainingDetection")) # ok
   
    runner = unittest.TextTestRunner()
    runner.run(suite)

