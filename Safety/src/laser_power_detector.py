#!/usr/bin/env python

'''

Laser power estimator based on different parameters

==================

Using IR image to estimate laser power.
It also can use on/off power modes to estimate background


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
from common import log, RectSelector

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

        self.first_time      = True   # help to deal with initialization
        self.frame_count     = 0      # count frames
        
        self.frame_gray      = None   # process separate channels
        self.frame_depth     = None   # process separate channels
        self.frame_left      = None   # raw video left frame
        self.frame_right     = None   # raw video right frame
        self.frame_show      = None   # show video frame
        self.figure_name     = 'Input'# figure name to show

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

    def get_frame(self, roi_type = 0):
        "get a single frame from the stream"
        # as the video frame.
        ret, frame              = self.video_src.read()  
        if not ret:
            return ret, []
        
        # # first time - inint some stuff
        # if self.first_time:
        #     self.frame_size     = frame.shape[:2]
        #     self.roi            = self.init_roi(roi_type)
        
        # # reduce image size
        # x0,y0,x1,y1             = self.roi
        # frame                   = frame[y0:y1,x0:x1,:]
                
        # convert channels
        ret                 = self.convert_frame_from_input(frame)

        # # number of channels
        # if self.first_time:    
        #     self.channel_num = 1 if len(frame_out.shape) < 3 else frame_out.shape[2]

        #self.frame_gray      = frame_gray.astype(np.float32)   # process separate channels
        #self.frame_depth     = frame_depth.astype(np.float32)   # process separate channels  
        frame_out            = frame
        self.first_time      = False
        self.frame_count     = self.frame_count + 1
        return True, frame_out.astype(np.float32)   
    
    def get_data(self):
        "get all the data strcutures"
        #self.first_time     = True
        #self.frame_count    = 0
        #self.integration_enb = True   # enable integration process

        ret, frame          = self.get_frame()  
        if not ret:
            self.tprint(f'Video source is not found ','E')
            return ret

        #self.frame_gray_int  = self.frame_gray   # process separate channels
        #self.frame_depth_int = self.frame_depth   # process separate channels

        #self.tprint('Detector is initialized')
        return ret    

    def show_data(self):
        "draw relevant image data"
        if self.frame_left is None or self.frame_right is None:
            self.tprint('No images found')
            return False
            
        # deal with black and white
        img_show = np.concatenate((self.frame_left, self.frame_right), axis = 1)

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
#%% Main
class LaserPowerEstimator:

    def __init__(self, estimator_type = 1, estimator_id = 0):

        # params
        # self.MIN_OBJECT_SIZE = 32       # minimal size of the object
        # self.MIN_STD_ERROR   = 10       # noise per pixel
        # self.MIN_VALID_DEPTH = 10       # pixel value below which the depth is not valid
        # self.BACKGROUND_NUM  = 3        # number of different background to estimate

        # show / debug
        self.idx             = estimator_id         # figure name to show        
        self.estimator_type  = estimator_type        # which type of the estimation to use
        self.estimator_options = {1:'std',2:'std integrated',11:'percent 30',12:'percent 10', 21:'Laser on-off'}

        
        #self.channel_num     = 1      # how many channels are processed - depends on the mode  
        self.frame_gray      = None   # process separate channels
        self.frame_depth     = None   # process separate channels
        self.frame_in        = None   # raw video frame
        self.frame_show      = None   # show video frame        
        self.frame_left      = None   # left IR video frame
        self.frame_right     = None   # right IR video frame        


        # integration over time
        self.frame_gray_int  = None   # process separate channels
        self.frame_depth_int = None   # process separate channels
        self.integration_enb = True   # enable integration process
        self.show_data_int   = False  # swicth that controls which dta to show
        self.img_int_mean    = None   # mean integrated image
        self.img_int_std     = None   # std integrated image

        # position and status
        self.img             = None   # original roi image patch
        self.pos             = (1,1)  # center of the roi
        self.size            = (10,10)# size of the roi
        self.good            = True   # psr above some threshold
        self.psr             = 0      # laser power / noise ratio
        self.rect            = None   # adjusted rectangle
        self.img_dbg         = None   # debug  image

        self.tprint(f'New power estimator type {estimator_type} and id {estimator_id} is defined')

    def init_roi(self, rect):
        "init ROI center and size and make it fft friendly"
        x1, y1, x2, y2  = rect
        w, h            = map(cv.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1          = (x1+x2-w)//2, (y1+y2-h)//2
        
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size       = w, h
        self.win        = cv.createHanningWindow((w, h), cv.CV_32F)

        (x, y), (w, h)  = self.pos, self.size
        x1, y1, x2, y2  = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        self.rect       = x1, y1, x2, y2
        self.tprint('ROI created %s' %str(rect)) 
        return True

    
    def preprocess(self, img):
        "image preprocessing - extracts roi and converts from uint8 to float using log function"

        x0, y0, x1, y1 = self.rect
        img_roi        = img[y0:y1,x0:x1].astype(np.float32)

        #img            = img.astype(np.float32)
        #img            = np.log(img + 1.0) 
        #img            = (img-img.mean()) / (img.std()+1e-9)
        img_roi           = img_roi #*self.win
        return img_roi    
    
    def estimate_percentile_contrast(self, img_roi, percent = 0.1):
        "estimate percentile of ythe ROI. Mean on low and max percent of the image"
        assert percent > 0 and percent < 0.5 , 'percentile must be between 0 and 0.5'
        low_val             = np.percentile(img_roi, percent * 100)
        high_val            = np.percentile(img_roi, (1-percent) * 100)
        psnr                = (high_val - low_val)/(high_val + low_val + 1e-9)
        return psnr    
    
    def estimate_image_psnr(self, img_roi):
        "makes  over ROI and estimaytes roi std. Mean/STD is an output"
        err_std             = img_roi.std() + 1e-9
        psnr                = img_roi.mean()/err_std
        return psnr    

    def estimate_integrated_image_psnr(self, img_roi, rate = 0.1):
        "makes integration over ROI in time and estimaytes roi std. Mean/STD is an output"
        err_std = 1e6

        if self.start_over or (self.img_int_mean is None):
            self.img_int_mean = img_roi
            self.img_int_std  = np.zeros_like(img_roi)
            self.start_over = False
        
        self.img_int_mean  += rate*(img_roi - self.img_int_mean)
        self.img_int_std   += rate*(np.abs(img_roi - self.img_int_mean) - self.img_int_std)
        err_std             = self.img_int_std.mean() + 1e-9
        psnr                = img_roi.mean()/err_std
        self.img_dbg        = self.img_int_std  # debug
        return psnr  

    def estimate_with_pattern_switch(self, img_roi):
        "assumes one image has pattern and other is not. The order is unknown"

        # first time
        if self.img_int_mean is None:
            self.img_int_mean = img_roi
        
        img_roi_prev        = self.img_int_mean
        img_diff            = img_roi - img_roi_prev

        psnr                = self.estimate_percentile_contrast(np.abs(img_diff))
        self.tprint(f'ROI PSNR : {psnr:.2f} ')
        self.img_dbg        = img_diff  # debug
        return psnr 

    def update(self, frame, rate = 0.1):
        "select differnet estimation techniques"
        ret     = False        
        if self.rect is None:
            self.tprint('Define ROI')
            return ret
        
        img_roi     = self.preprocess(frame)

        if self.estimator_type == 1: # simple mean / std

            psnr        = self.estimate_image_psnr(img_roi)

        elif self.estimator_type == 2: # simple mean / std        
        
            psnr        = self.estimate_integrated_image_psnr(img_roi, rate)

        elif self.estimator_type == 11: # percentile        
        
            psnr        = self.estimate_percentile_contrast(img_roi, 0.3) 

        elif self.estimator_type == 12: # percentile        
        
            psnr        = self.estimate_percentile_contrast(img_roi, 0.1)  

        elif self.estimator_type == 21: # on-off switch        
        
            psnr        = self.estimate_with_pattern_switch(img_roi)                                   

        self.psr = psnr
        self.img = img_roi   # for debug
        return True
    
    def convert_frame_for_show(self, frame = None):
        "converts different frame types to the uint8 3 colors"

        if self.show_data_int:
            frame_g, frame_d = self.frame_gray_int, self.frame_depth_int
        else:
            frame_g, frame_d = self.frame_gray, self.frame_depth


        if self.mode == 'rgb':
            img_show    = np.uint8(frame_g) #.copy()
            #img_show    = cv.applyColorMap(img_show, cv.COLORMAP_JET)
        elif self.mode == 'ddd':
            img_show    = np.uint8(frame_d)
            #img_show    = cv.applyColorMap(img_show, cv.COLORMAP_JET)
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
    
    def show_internal_state(self):
        "shows some internal info"
        imgin           = np.uint8(self.img)
        f               = self.img - self.img.mean()
        kernel          = np.uint8((f-f.min()) / f.ptp()*255 )
        r               = self.img_dbg
        resp            = np.uint8((r-r.min()) / r.ptp()*255) #np.clip(resp/resp.max(), 0, 1)*255)
        vis             = np.hstack([imgin, kernel, resp])

        figure_name     = f'{self.estimator_type} - {self.idx}'        
        cv.imshow(figure_name, vis)
        return True    

    def show_state(self, vis):
        # show state of the estimator
        if self.rect is None:
            return vis
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        else:
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        vis  = draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)   
        return vis 

    def show_scene(self, frame):
        "draw scene and ROI"
        if len(frame.shape) > 2:
            vis     = frame.copy()
        else:
            vis     = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)

        # rectangle state
        vis             = self.show_state(vis)

        # type of the estimator
        estimator_name  = self.estimator_options[self.estimator_type]
        figure_name     = f'{estimator_name} - {self.idx}'
        cv.imshow(figure_name, vis)
        ch  = cv.waitKey(30)
        ret = ch != ord('q')     

        return ret                

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

    def run_show_video(self, figure_name = 'Input', roi_list = []):
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

    def finish(self):
        # Close down the video stream
        #self.video_src.release()
        cv.destroyAllWindows()

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
class TestPowerEstimator(unittest.TestCase):

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

    def test_video_std(self):
        "show video and measure std over time"
        d       = DataSource()
        p       = LaserPowerEstimator(1)
        srcid   = 11
        rect    = (280,200,360,280)
        ret     = d.init_video(srcid)
        retp    = p.init_roi(rect)
        while ret:
            retd    = d.get_data()
            rets    = d.show_data() 
            retp    = p.update(d.frame_left)
            ret     = p.show_scene(d.frame_left) and retd

        #p.run_video_integration(str(srcid))
        p.finish()
        self.assertTrue(not ret) 

    def test_video_percentile(self):
        "show video and measure infrared roi percentile"
        d       = DataSource()
        p       = LaserPowerEstimator(12)
        srcid   = 11
        rect    = (280,200,360,280)
        ret     = d.init_video(srcid)
        retp    = p.init_roi(rect)
        while ret:
            retd    = d.get_data()
            rets    = d.show_data() 
            retp    = p.update(d.frame_left)
            ret     = p.show_scene(d.frame_left) and retd

        #p.run_video_integration(str(srcid))
        p.finish()
        self.assertTrue(not ret)         

    def test_video_integration(self):
        "show video and integrate over time"
        d       = DataSource()
        p       = LaserPowerEstimator()
        srcid   = 11
        rect    = (280,200,360,280)
        ret     = d.init_video(srcid)
        retp    = p.init_roi(rect)
        while ret:
            ret     = d.get_data()
            ret     = d.show_data() and ret
            retp    = p.update(d.frame_left)

        #p.run_video_integration(str(srcid))
        p.finish()
        self.assertTrue(not ret) 

    def test_video_with_pattern_switch(self):
        "show video with pattern on off each frame"
        srcid   = 21       # video tytpe
        rect    = (280,200,360,280)
        estid   = 21       # estimator id

        d       = DataSource()
        p       = LaserPowerEstimator(estimator_type = estid)
        ret     = d.init_video(srcid)
        retp    = p.init_roi(rect)
        while ret:
            ret     = d.get_data()
            ret     = d.show_data() and ret
            retp    = p.update(d.frame_left)
            ret     = p.show_scene(d.frame_left) and ret
            reti    = p.show_internal_state()

        p.finish()
        self.assertTrue(not ret)         


# --------------------------------
#%% App
class App:
    def __init__(self, video_src = 'ii2'):

        #self.cap = video.create_capture(video_src)
        self.cap        = RealSense(video_src)
        self.cap.change_mode('iig')

        _, self.frame   = self.cap.read()
        frame_gray      = self.get_frame(0)
        vis             = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR) 
        self.imshow_name= 'Power Detect (u-Update, p-Project, space-Pause, q-Quit)'
        cv.imshow(self.imshow_name, vis)
        self.rect_sel   = RectSelector(self.imshow_name, self.on_rect)
        self.trackers   = []
        self.paused     = False
        self.update_rate= 0 
        self.estimator_options = {1:'std',2:'std integrated',11:'percent 30',12:'percent 10'}

    def get_frame(self, frame_type = 0):
        "extracts gray frame"
        h, w = self.frame.shape[:2]
        if len(self.frame.shape) < 3:
            frame_gray = self.frame
        elif frame_type == 0:
            frame_gray = self.frame[:,:,0]
        elif frame_type == 1:
            frame_gray = self.frame[:,:,1]            
        else:
            frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

        # if two frames are concatenated - take one
        w2 = w>>1
        if h < w2:
            frame_gray = frame_gray[:,:w2]

        return frame_gray                  

    def on_rect(self, rect):
        #frame_gray      = self.frame #cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        estim_ind       = len(self.trackers) + 1
        estim_type      = 12
        tracker         = LaserPowerEstimator(estimator_type=estim_type, estimator_id=estim_ind)
        #frame_gray      = self.get_frame(0)
        tracker.init_roi(rect)
        self.trackers.append(tracker)

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break

            frame_gray  = self.get_frame(0)
            for tracker in self.trackers:
                tracker.update(frame_gray, rate = self.update_rate)                 

            #vis = self.frame.copy()
            vis = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR) 
            for tracker in self.trackers:
                tracker.show_state(vis)

            #if len(self.trackers) > 0:
            #    cv.imshow('tracker state', self.trackers[-1].state_vis)                

            self.rect_sel.draw(vis) # draw rectangle
            cv.imshow(self.imshow_name, vis)
            ch = cv.waitKey(1)
            if ch == 27 or ch == ord('q'):
                break            
            elif ch == ord(' '):
                self.paused = not self.paused
            elif ch == ord('c'):
                self.trackers.pop()
            elif ch == ord('u'):  
                self.update_rate = 0.1 if self.update_rate < 0.001 else 0  
            elif ch == ord('p'):
                self.cap.use_projector = not self.cap.use_projector
                self.cap.switch_projector()                                


if __name__ == '__main__':
    #print(__doc__)

    """ """
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestPowerEstimator("test_data_source_rs")) # ok
    #suite.addTest(TestPowerEstimator("test_data_source_video")) # ok
    #suite.addTest(TestPowerEstimator("test_video_std")) # ok
    #suite.addTest(TestPowerEstimator("test_video_percentile")) # ok
    suite.addTest(TestPowerEstimator("test_video_with_pattern_switch")) 
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
    

    #App('iig').run()    

