## Works with old sensor.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

class RealSense(object):
    def __init__(self, size=None, mode = 'rgb', **params):
        
        self.frame_size = (1280, 720)
        self.count = 0
        # if size is not None:
        #     w, h = map(int, size.split('x'))
        #     self.frame_size = (w, h)
        #     #self.bg = cv.resize(self.bg, self.frame_size)    
        #     # 
        self.mode = 'rgb' if mode is None else mode   

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #print(rs.__version__)
        #self.pipe = rs.pipeline()
        #self.cfg = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        #self.pipe.start(self.cfg)

        #     # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = self.config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        #found_rgb = False
        # for s in device.sensors:
        #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
        #         found_rgb = True
        #         break
        # if not found_rgb:
        #     print("The demo requires Depth camera with Color sensor")
        #     exit(0)       


   

        # self.config.enable_stream(rs.stream.depth, self.frame_size[0], self.frame_size[1], rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, self.frame_size[0], self.frame_size[1], rs.format.bgr8, 30)
        # #self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # #self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # Start streaming
        profile = self.pipeline.start(self.config)

        

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def render(self, dst):
        pass

    def change_mode(self, mode = 'rgb'):
        if not(mode in ['rgb','rgd','gd','ddd','ggd','gdd','scl','dep']):
               print(f'Not supported mode = {mode}')
               #return
        self.mode = mode  

    def read(self, dst=None):
        "with frame alignments"
        w, h = self.frame_size

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #color_image = cv.cvtColor(depth_image, cv.COLOR_GRAY2RGB)
        #depth_image = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap  = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        #If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)
            #images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # images = color_image

        if self.mode == 'rgb':
            image_out = color_image
        elif self.mode == 'ddd':
            image_out = depth_colormap
        elif self.mode == 'rgd':
            image_out = np.concatenate((color_image[:,:,:2], depth_scaled[:,:,np.newaxis] ), axis = 2)
        elif self.mode == 'gd':
            gray_image  = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out = np.concatenate((gray_image, depth_scaled ), axis = 1)
        elif self.mode == 'ggd':
            gray_image  = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out = np.stack((gray_image, gray_image, depth_scaled ), axis = 2)            
        elif self.mode == 'gdd':
            gray_image  = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out = np.stack((gray_image, depth_scaled, depth_scaled ), axis = 2) 
        elif self.mode == 'scl':
            depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.1)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)    
        elif self.mode == 'dep':
            image_out  = depth_image                    
        return True, image_out

    def read__not_aligned(self, dst=None):
        "color and depth are not aligned"
        w, h = self.frame_size

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #color_image = cv.cvtColor(depth_image, cv.COLOR_GRAY2RGB)
        #depth_image = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap  = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        #If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)
            #images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # images = color_image

        if self.mode == 'rgb':
            image_out = color_image
        elif self.mode == 'ddd':
            image_out = depth_colormap
        elif self.mode == 'rgd':
            image_out = np.concatenate((color_image[:,:,:2], depth_scaled[:,:,np.newaxis] ), axis = 2)
        elif self.mode == 'gd':
            gray_image  = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out = np.concatenate((gray_image, depth_scaled ), axis = 1)
        elif self.mode == 'ggd':
            gray_image  = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out = np.stack((gray_image, gray_image, depth_scaled ), axis = 2)            
        elif self.mode == 'gdd':
            gray_image  = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out = np.stack((gray_image, depth_scaled, depth_scaled ), axis = 2) 
        elif self.mode == 'scl':
            depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.1)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)    
        elif self.mode == 'dep':
            image_out  = depth_image                    
        return True, image_out

    def isOpened(self):
        return True
    
    def save_image(self, frame):
        fn = '.\image_%s_%03d.png' % (self.mode, self.count)
        cv.imwrite(fn, frame)
        print(fn, 'saved')
        self.count += 1   
    
    def close(self):
        # Stop streaming
        self.pipeline.stop()

    def test(self):
        while True:
            ret, frame = self.read()
            if ret is False:
                break
        
            cv.imshow('frame (c,d,1,g,s,f,h: q - to exit)', frame)
            ch = cv.waitKey(10) & 0xff
            if ch == ord('q'):
                break
            elif ch == ord('c'):
                self.change_mode('rgb')
            elif ch == ord('d'):
                self.change_mode('ddd')            
            elif ch == ord('g'):
                self.change_mode('gd') 
            elif ch == ord('b'):
                self.change_mode('rgd') 
            elif ch == ord('f'):
                self.change_mode('ggd') 
            elif ch == ord('a'):
                self.change_mode('gdd')     
            elif ch == ord('1'):
                self.change_mode('scl')                            
            elif ch == ord('h'):
                self.change_mode('dep') 
            elif ch == ord('s'):
                self.save_image(frame) 

        if ret is False:
            print('Failed to read image')
        else:
            self.close()
        cv.destroyAllWindows()

if __name__ == '__main__':
    cap = RealSense()
    cap.test()