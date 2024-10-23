
'''
OpenCV like wrapper for Real Sense Camera

==================

Allows to read, display store video and images of RGB - Depth combinations in different formats.  
Can extract left and right IR images.
Aligns RGB and Depth data.
Can save data as mp4 or single images.

Usage:
    python opencv_realsense_camera.py 
    will run the camera and open the image window with live stream.
    Use keys outlines in test() function to switch different modes
    Press 's' to save the current image
    Press 't' to save the left and right images in separate files
    Press 'r' to start recording and one more time 'r' to stop video recording
    
                                        

Environment : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 


'''
import os
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time

class RealSense(object):
    def __init__(self,  mode = 'rgb', use_ir = True, **params):
        
        self.frame_size     = (640,480) #(1280, 720)#
        self.count          = 0
        self.mode           = 'rgb' if mode is None else mode 
        self.use_ir         = False if use_ir is None else use_ir
        self.use_projector  = False

        # Configure depth and color streams
        self.pipeline       = rs.pipeline()
        self.config         = rs.config()

        self.config.enable_stream(rs.stream.depth, self.frame_size[0], self.frame_size[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.frame_size[0], self.frame_size[1], rs.format.bgr8, 30)
        
        if self.use_ir:
            self.config.enable_stream(rs.stream.infrared, 1)
            self.config.enable_stream(rs.stream.infrared, 2)
            print('IR is enabled')
        else:
            print('IR is disabled')

        #  Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        try: # 545
            pipeline_profile = self.config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))
            device_name = device.get_info(rs.camera_info.name)
            print('Device name : ', device_name)
            print('Device product line : ', device_product_line)
        except Exception as e:
            print('Real Sense new version - possibly will require a new driver version')
            print(e)


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
        profile             = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor   = profile.get_device().first_depth_sensor()
        depth_scale         = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # # turn emitter on-off
        # if device_product_line != 'D400':
        #     depth_sensor.set_option(rs.option.emitter_enabled, 0)
        # check features
        self.has_projector       = device_name.find('D455') > 0
        # if not self.has_projector:
        #     print('Camera without projector')
        # else:
        #     if self.use_projector is False:
        #         self.depth_sensor.set_option(rs.option.emitter_always_on, False)
        #         self.depth_sensor.set_option(rs.option.emitter_enabled, False)
                
        #     print('Camera projector : %s' %str(self.use_projector))     
        self.switch_projector()   

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to        = rs.stream.color
        self.align      = rs.align(align_to)

        # record video
        self.vout       = None
        self.record_on  = False # toggle recording

    def render(self, dst):
        pass

    def switch_projector(self):
        "switch projectpr on-off"        
        if not self.has_projector:
            print('Camera is without projector')
        else:
            #if self.use_projector is False:
            self.depth_sensor.set_option(rs.option.emitter_always_on, self.use_projector)
            self.depth_sensor.set_option(rs.option.emitter_enabled, self.use_projector)
                
            time.sleep(0.1) # wait for camera on - off
            print('Camera projector : %s' %str(self.use_projector)) 
            

    def change_mode(self, mode = 'rgb'):
        if not(mode in ['rgb','rgd','gd','ddd','ggd','gdd','scl','sc2','dep','iid','ii2','iig','iir']):
             print(f'Not supported mode = {mode}')
               
        self.mode = mode  
        print(f'Current mode {mode}')


    def read(self, dst=None):
        "with frame alignments and color space transformations"
        #self.use_projector = not self.use_projector # testing
        w, h                = self.frame_size

        # Wait for a coherent pair of frames: depth and color
        frames              = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames      = self.align.process(frames)

        # Get aligned frames
        depth_frame         = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame         = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None

        # Convert images to numpy arrays
        depth_image         = np.asanyarray(depth_frame.get_data())
        color_image         = np.asanyarray(color_frame.get_data())
        #color_image = cv.cvtColor(depth_image, cv.COLOR_GRAY2RGB)
        #depth_image = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_scaled        = cv.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap      = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        #If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)
            #images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        if self.use_ir:
            ir_left     = aligned_frames.get_infrared_frame(1)
            irl_image   = np.asanyarray(ir_left.get_data())
            ir_right    = aligned_frames.get_infrared_frame(2)
            irr_image   = np.asanyarray(ir_right.get_data())
        else:
            #print('Enable IR use at the start. use_ir = True')    
            irl_image   = color_image[:,:,0]
            irr_image   = color_image[:,:,1]
            image_out   = color_image            

        if self.mode == 'rgb':
            image_out       = color_image
        elif self.mode == 'ddd':
            image_out       = depth_colormap
        elif self.mode == 'rgd':
            image_out       = np.concatenate((color_image[:,:,:2], depth_scaled[:,:,np.newaxis] ), axis = 2)
        elif self.mode == 'gd':
            gray_image      = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out       = np.concatenate((gray_image, depth_scaled ), axis = 1)
        elif self.mode == 'ggd':
            gray_image      = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out       = np.stack((gray_image, gray_image, depth_scaled ), axis = 2)            
        elif self.mode == 'gdd':
            gray_image      = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
            image_out       = np.stack((gray_image, depth_scaled, depth_scaled ), axis = 2) 
        elif self.mode == 'scl':
            depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.05)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)    
        elif self.mode == 'sc2':
            depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.1)
            image_out       = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)    
        elif self.mode == 'ii2':
            image_out       = np.concatenate((irl_image, irr_image), axis = 1)  
        elif self.mode == 'iid':
            image_out       = np.stack((irl_image, irr_image, depth_scaled), axis = 2)  
            #image_out       = np.concatenate((irl_image, depth_scaled), axis = 1) 
        elif self.mode == 'iig':
            image_out       = np.stack((irl_image, irr_image, color_image[:,:,1]), axis = 2)                  
        elif self.mode == 'iir':
            image_out       = np.stack((irl_image, irr_image, color_image[:,:,0]), axis = 2) 
            #image_out       = np.concatenate((irl_image, color_image[:,:,0]), axis = 1) 
        elif self.mode == 'dep':
            image_out       = depth_image                    
        return True, image_out

    def read_not_aligned(self, dst=None):
        "color and depth are not aligned"
        w, h = self.frame_size

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None

        # Convert images to numpy arrays
        depth_image     = np.asanyarray(depth_frame.get_data())
        color_image     = np.asanyarray(color_frame.get_data())
        #color_image = cv.cvtColor(depth_image, cv.COLOR_GRAY2RGB)
        #depth_image = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_scaled    = cv.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap  = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        image_out  = depth_image 
        return True, image_out


    def isOpened(self):
        "OpenCV compatability"
        return True
    
    def save_image(self, frame):
        fn = '.\\image_%s_%03d.png' % (self.mode, self.count)
        cv.imwrite(fn, frame)
        print(fn, 'saved')
        self.count += 1   

    def save_two_images(self, frame):
        fl = '.\\imageL_%s_%03d.png' % (self.mode, self.count)
        cv.imwrite(fl, frame[:,:,0])
        fr = '.\\imageR_%s_%03d.png' % (self.mode, self.count)
        cv.imwrite(fr, frame[:,:,1])
        print('Saving %s and %s' %(fl,fr))
        self.count += 1          

    def record_video(self, frame):
        # record video to a file is switched on
        if (self.vout is None) and (self.record_on is True):
            fourcc  = cv.VideoWriter_fourcc(*'mp4v')
            k       = 0
            fname   = '.\\video_%s_%03d.mp4' % (self.mode,k)
            while os.path.exists(fname):
                k      +=1
                fname   = '.\\video_%s_%03d.mp4' % (self.mode,k)

            self.vout     = cv.VideoWriter(fname, fourcc, 20.0, self.frame_size)
            print('Writing video to file %s' %fname)
            self.count = 0

        # write frame
        if (self.vout is not None) and (self.record_on is True):
            ""
            if len(frame.shape) < 3:
                frame = frame[:self.frame_size[1],:self.frame_size[0]]
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)

            self.vout.write(frame)
            self.count += 1  
            if self.count % 100 == 0:
                print('Writing frame %s' %str(self.count))

        # record on is switched off
        if (self.vout is not None) and (self.record_on is False):
            self.vout.release()
            self.vout = None
            print('Video file created')

    def record_release(self):
        "finish record"         
        if self.vout is not None:
            self.vout.release()
            self.vout = None
            print('Video file created')

    def show_image(self, frame):
        "show image on opencv window"
        do_exit = False
        cv.imshow('frame (c,a,d,1,2,g,s,t,f,h,k,j,i,o,p,r: q - to exit)', frame)
        ch = cv.waitKey(1) & 0xff
        if ch == ord('q') or ch == 27:
            do_exit = True 
        elif ch == ord('c'): # regular RGB image
            self.change_mode('rgb')
        elif ch == ord('d'): # depth image
            self.change_mode('ddd')            
        elif ch == ord('g'): # concatenated g and d
            self.change_mode('gd') 
        elif ch == ord('b'):
            self.change_mode('rgd') 
        elif ch == ord('f'):
            self.change_mode('ggd') 
        elif ch == ord('a'):
            self.change_mode('gdd')     
        elif ch == ord('1'):
            self.change_mode('scl')  
        elif ch == ord('2'):
            self.change_mode('sc2')                                           
        elif ch == ord('i'):
            self.change_mode('ii2') 
        elif ch == ord('o'):
            self.change_mode('iid') 
        elif ch == ord('k'):
            self.change_mode('iig')    
        elif ch == ord('j'):
            self.change_mode('iir')                    
        elif ch == ord('h'):
            self.change_mode('dep')                 
        elif ch == ord('s'):
            self.save_image(frame) 
        elif ch == ord('t'):
            self.save_two_images(frame)             
        elif ch == ord('r'):
            self.record_on = not self.record_on
            print('Video record %s' %str(self.record_on))
        elif ch == ord('p'):
            self.use_projector = not self.use_projector
            self.switch_projector()
        elif ch != 255:
            print('unrecognized key - check your language setttings on the keyboard, must be English.')
        return do_exit
          
    def close(self):
        # stop record
        self.record_release()

        # Stop streaming
        self.pipeline.stop()

    def release(self):
        "opencv compatability"
        self.close()

    def test(self):
        while True:
            ret, frame = self.read()
            if ret is False:
                break
        
            ret     = self.show_image(frame)
            if ret :
                break  

            # check if record is required
            self.record_video(frame)   

        if ret is False:
            print('Failed to read image')
        else:
            self.close()
        cv.destroyAllWindows()

if __name__ == '__main__':
    cap = RealSense()
    cap.test()