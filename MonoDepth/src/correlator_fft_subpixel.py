
"""
Subpixel correlation using FFT

Environment : 

Installation:

Usage:


-----------------------------
 Ver    Date     Who    Descr
-----------------------------
0101   18.09.24 UD     Started. 
-----------------------------
"""


import numpy as np
import unittest
import cv2 as cv
import matplotlib.pyplot as plt

from scipy.fftpack import fft2, ifft2
from scipy.signal.windows import triang
from numpy import unravel_index

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from opencv_realsense_camera import RealSense
from common import log, RectSelector

sys.path.append(r'..\MonoDepth\src')
from peak_fit_2d import peak_fit_2d


eps = 1e-6
# ----------------------
#%% Helper function
def max_location(a):
    "position in 2d array"
    return unravel_index(a.argmax(), a.shape)

# ----------------------
#%% Generate image pairs and points
class DataGenerator:
    "class to create images and correspondance points"
    def __init__(self):

        self.frame_size = (640,480)
        self.imgD        = None
        self.imgL        = None
        self.imgR        = None

    def add_noise(self, img_gray, noise_percentage = 0.01):
        "salt and pepper noise"
        if noise_percentage < 0.001:
            return img_gray


        # Get the image size (number of pixels in the image).
        img_size = img_gray.size

        # Set the percentage of pixels that should contain noise
        #noise_percentage = 0.1  # Setting to 10%

        # Determine the size of the noise based on the noise precentage
        noise_size = int(noise_percentage*img_size)

        # Randomly select indices for adding noise.
        random_indices = np.random.choice(img_size, noise_size)

        # Create a copy of the original image that serves as a template for the noised image.
        img_noised = img_gray.copy()

        # Create a noise list with random placements of min and max values of the image pixels.
        #noise = np.random.choice([img_gray.min(), img_gray.max()], noise_size)
        noise = np.random.choice([-10, 10], noise_size)

        # Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
        img_noised.flat[random_indices] += noise
        
        self.tprint('adding image noise')
        return img_noised

    def init_roi(self, test_type = 0):
        "load the roi case"
        w,h     = self.frame_size
        w2, h2  = w>>1, h>>1
        roi     = [0,0,w,h]
        if test_type == 1:
            roi = [w2-3,h2-3,w2+3,h2+3] # xlu, ylu, xrb, yrb
        elif test_type == 2:
            roi = [300,220,340,260] # xlu, ylu, xrb, yrb
        elif test_type == 3:
            roi = [280,200,360,280] # xlu, ylu, xrb, yrb            
        elif test_type == 4:
            roi = [220,140,420,340] # xlu, ylu, xrb, yrb      
        elif test_type == 5:
            roi = [200,120,440,360] # xlu, ylu, xrb, yrb     
   
        return roi    
    
    def init_image(self, img_type = 1, window_size = 32):
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
            self.imgL = np.roll(self.imgL[100:400,100:600], np.array([0, 0]), axis=(0, 1)) 
            self.imgR = self.imgR[100:400,100:600]        

        elif img_type == 5:
            self.imgD = cv.pyrDown(cv.imread(r"C:\Data\Corr\d2_Depth.png", cv.IMREAD_GRAYSCALE))
            self.imgL = cv.pyrDown(cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE))
            self.imgR = cv.pyrDown(cv.imread(r"C:\Data\Corr\r3_Infrared.png", cv.IMREAD_GRAYSCALE) )             

        elif img_type == 6: # same image with shift
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)
            self.imgR = np.roll(self.imgL, np.array([-4, -4]), axis=(0, 1))

        elif img_type == 7: # same image with shift - sub region
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)[100:328, 300:528]
            self.imgR = np.roll(self.imgL, np.array([-4, -4]), axis=(0, 1))

        elif img_type == 11:  # Test patterns for correlation no scaling
            scale       = 1
            shift       = np.array([1, 1]) * 2

            image1      = np.random.rand(32, 32) * 60 + 0    
            image1_hr   = image1
            image1_hr   = np.roll(image1_hr, shift, axis=(0, 1))  
            image2      = image1_hr[::scale,::scale]

            self.imgL, self.imgR = image1, image2

        elif img_type == 12:  # Test pattern against image - in sync
            scale       = 4
            shift       = np.array([1, 1]) * 3

            image1      = np.random.rand(32, 32) * 60 + 50
            size_orig   = image1.shape[::-1]
            size_hr     = (size_orig[0]*scale , size_orig[1]*scale)        
            image1_hr   = cv.resize(image1, size_hr ,cv.INTER_CUBIC)
            image1_hr   = np.roll(image1_hr, shift, axis=(0, 1))  
            #image2      = image1_hr[::scale,::scale]
            image2      = cv.resize(image1_hr, size_orig ,cv.INTER_CUBIC)

            self.imgL, self.imgR = image1, image2

        elif img_type == 13:  # Test pattern against image - half win shift
            image1      = np.random.rand(128, 128) * 60 + 0
            offset      = 40
            image_patch = image1[offset:offset+window_size,offset:offset+window_size]
            image2      = np.tile(image_patch, (8, 8))
            shift       = np.array([2, 2]) * 0
            image1      = np.roll(image1, shift, axis=(0, 1))   
            self.imgL, self.imgR = image1, image2

        elif img_type == 14:  # Test pattern against image - half win shift
            image1      = np.random.rand(128, 128) * 60 + 0
            offset      = [40,32]
            image_patch = image1[offset[0]:offset[0]+window_size,offset[1]:offset[1]+window_size]
            image2      = np.tile(image_patch, (8, 8))
            shift       = np.array([2, 2]) * 0
            image1      = np.roll(image1, shift, axis=(0, 1))  
            self.imgL, self.imgR = image1, image2          

        elif img_type == 21:  # Test one image against image 
            image1      = np.random.rand(128, 128) * 60 + 60
            image2      = image1.copy()
            shift       = np.array([-5, -8])*1
            image1      = np.roll(image1, shift, axis=(0, 1))  
            self.imgL, self.imgR = image1, image2   

        elif img_type == 22:  # Test random image
            image1      = np.random.rand(128, 128) * 60 + 60
            shift       = np.array([4, 4]) * -1
            image2      = np.roll(image1, shift, axis=(0, 1))
            self.imgL, self.imgR = image1, image2

        elif img_type == 31:  # Test simple image
            # Assuming 'trees' is a predefined image
            image       = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_1004.png")
            image1      = cv.cvtColor(image, cv.COLOR_RGB2GRAY)      
            shift       = np.array([2, 2]) * 2
            image2      = np.roll(image1, shift, axis=(0, 1))
            self.imgL, self.imgR = image1, image2

        elif img_type == 32:  # Small size
            image1      = np.zeros((64, 64))
            image1[4 * window_size // 2 : 4 * window_size // 2 + window_size,
                    4 * window_size // 2 : 4 * window_size // 2 + window_size] = 100
            shift       = np.array([2, 2]) * 1
            image2      = np.roll(image1, shift, axis=(0, 1))
            self.imgL, self.imgR = image1, image2

        elif img_type == 34:  # Small size
            image1      = np.random.rand(40, 40)
            shift       = np.array([2, 2]) * 1
            image2      = np.roll(image1, shift, axis=(0, 1))

        elif img_type == 35:  # Test different size
            image1      = np.zeros((123, 171)) * 3
            image1[50:58, 50:58] = 180
            shift       = np.array([2, 2]) * 1
            image2      = np.roll(image1, shift, axis=(0, 1))
            self.imgL, self.imgR = image1, image2            

        else:
            raise ValueError("Unknown TestType")    
                      
        #self.imgC       = self.imgD    
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

    def show_images(self):
        "show images left and right"
        image_out   = np.concatenate((self.imgL, self.imgR), axis = 1) 
        image_out   = np.uint8(image_out)
        while image_out.shape[1] > 1999:
            image_out = cv.pyrDown(image_out)

        while image_out.shape[1] < 600:
            image_out = cv.pyrUp(image_out)

        cv.imshow('Input Images', image_out)
        ch = cv.waitKey(1) & 0xff

# ----------------------
#%% Motion Estimation
class CORR:
    """ 
    Uses 2D correlation for matching on a single window with sub pixel resolution
    """

    def __init__(self):

        self.frame_size     = (640,480)

    def check(self, img1, img2):
        "checking images"

        y1, x1          = img1.shape[:2]
        y2, x2          = img2.shape[:2]

        if y1 != y2 or x1 != x2:
            raise ValueError('images must be the same size')

        w, h            = map(cv.getOptimalDFTSize, [x2, y2])
        if h != y2 or w != x2:
            w,h         = w - 32,h - 32
            #raise ValueError('image sizes must be multiple of power 2')
            img1, img2  = img1[:h,:w], img2[:h,:w]

        self.size       = w, h
        self.win        = cv.createHanningWindow((w, h), cv.CV_32F)        

        return img1, img2

    def preprocess(self, img, corr_enabled = False):
        "image preprocessing - converts from uint8 to float using log function"
        img            = img.astype(np.float64)
        img            = np.log(img + 1.0) if corr_enabled else img
        #img            = (img-img.mean()) / (img.std()+1e-9)
        img           = img*self.win
        return img
    
    def frequency_shaping(self, A):
        "remove DC"
        A[:,:,0]   = A[:,:,0]*self.win
        A[:,:,1]   = A[:,:,1]*self.win
        return A
    
    def correlate(self, img1, img2):
        "do the correlation"
        img1, img2  = self.check(img1, img2)

        a           = self.preprocess(img1)
        A           = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
        A           = self.frequency_shaping(A)

        b           = self.preprocess(img2)
        B           = cv.dft(b, flags=cv.DFT_COMPLEX_OUTPUT)
        B           = self.frequency_shaping(B)

        C           = cv.mulSpectrums(A, B, 0, conjB=True)

        #respr       = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        #respr       = np.fft.fftshift(respr) # Shift the zero-frequency component to the center

        # UD  subpixel
        respc       = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_COMPLEX_OUTPUT)
        respc       = np.fft.fftshift(respc)
        resp        = cv.magnitude(respc[:,:,0],respc[:,:,1])   

        h, w        = resp.shape
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)

        cval        = respc[my,mx,:].squeeze()
        angl        = np.arctan2(cval[0],cval[1])*180/np.pi
           

        side_resp   = resp.copy()
        side_resp   = cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr         = (mval-smean) / (sstd+eps)

        # # sub pixel
        # search_size = 3
        # z           = resp[my-search_size:my+search_size, mx-search_size:mx+search_size]
        # xp, yp      = peak_fit_2d(z)
        # xp, yp      = xp - search_size, yp - search_size     
        # #mx, my      = mx + xp, my + yp
        # #print(f"{my:.2f},{mx:.2f}: {yp:.2f},{xp:.2f}")
        # print(f"{my:.2f},{mx:.2f}")
        # #time.sleep(0.5)
        xp, yp      = peak_fit_2d(side_resp)
        xp, yp      = xp - 5, yp - 5
        print(f'Peak subpixel : {xp:.2f}, {yp:.2f}')


        #side_resp       = resp.copy()
        #cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)    
        # 
        print(f"PSR {psr:.2f} : {my},{mx}: {cval[0]:.2f}, {cval[1]:.2f} : {angl:.2f}")        

        return resp
    
    def show_result(self, image_patch):
        # debug
        peak_xy    = max_location(np.abs(image_patch))
        plt.figure(21)
        plt.imshow(np.abs(image_patch), cmap='gray')
        plt.title('Peak at %s' %(str(peak_xy)))
        plt.colorbar() #orientation='horizontal')
        plt.show()

    def show_corr_image(self, correlation_image, fig_num = 31):
        "shows the correlation image"
        peak_xy           = max_location(np.abs(correlation_image))
        peak_val          = correlation_image.max()
        txts              = 'Max val : %s, X,Y = : %s,%s' %(str(peak_val), str(peak_xy[0]), str(peak_xy[1]))
        self.tprint(txts)

        # Visualize correlation
        plt.figure(fig_num)
        #plt.imshow(np.log10(np.abs(correlation_image)), cmap='gray')
        plt.imshow(np.abs(correlation_image), cmap='gray')
        #plt.title(f"Correlation Shift: {shift}")
        plt.title(txts)
        plt.colorbar() #orientation='horizontal')
        plt.show()  

    def tprint(self, ptxt='',level='I'):
        
        if level == 'I':
            #ptxt = 'I: STF: %s' % txt
            log.info(ptxt)  
        if level == 'W':
            #ptxt = 'W: STF: %s' % txt
            log.warning(ptxt)  
        if level == 'E':
            #ptxt = 'E: STF: %s' % txt
            log.error(ptxt)  
           
        #print(ptxt)

# ----------------------
#%% Tests
class TestCORR(unittest.TestCase):


    def test_corr(self):
        "correlator of left and right random images - the same size"

        d       = DataGenerator()
        isOk    = d.init_image(img_type = 11)
        d.show_images()

        p       = CORR()
        img_c   = p.correlate(d.imgL, d.imgR)

        xp, yp  = peak_fit_2d(img_c)
        print(f'Peak subpixel : {xp:.2f}, {yp:.2f}')

        p.show_corr_image(img_c)
        self.assertTrue(isOk)       

    def test_corr_interpolated(self):
        "correlator of left and right random images - with interpolation"

        d       = DataGenerator()
        isOk    = d.init_image(img_type = 4)  # 12-ok, 6,7-ok, 5-ok
        d.show_images()

        p       = CORR()
        img_c   = p.correlate(d.imgL, d.imgR)

        xp, yp  = peak_fit_2d(img_c)
        print(f'Peak subpixel : {xp:.2f}, {yp:.2f}')

        p.show_corr_image(img_c)
        self.assertTrue(isOk)  

    def test_1d(self):
        "self check"
        F   = 7
        phi = np.deg2rad(40)

        t   = np.linspace(0, 1, 100)
        s1  = np.sin(F*np.pi*2*t + phi) #+ 1j*t*0
        f1  = np.fft.fft(s1)

        s2  = np.sin(F*np.pi*2*t) #+ 1j*t*0
        f2  = np.fft.fft(s2)

        fc  = f1*np.conj(f2)
        c   = np.fft.ifft(fc)
        ca  = np.abs(c)

        ip   = np.where(ca==ca.max())[0]
        phir = np.rad2deg(np.angle(c[ip]))
        
        print(phir)

        import matplotlib.pyplot as plt
        #plt.figure()
        plt.plot(t,s1,'g',t,s2,'b',t,ca/10,'r')
        plt.show()

            

# ----------------------
#%% App
class App:
    def __init__(self):
        self.cap   = RealSense()
        self.cap.change_mode('iid')

        self.corr = CORR()

        self.frame  = None
        self.paused = False

        cv.namedWindow('Corr')

    def run(self):
        while True:
            playing = not self.paused
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            #vis = self.frame.copy()
            #if playing:
            img_c   = self.corr.test_CORR_corr(self.frame[:,:,0], self.frame[:,:,1], window_size = 16)
                # for tr in tracked:
                #     cv.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                #     for (x, y) in np.int32(tr.p1):
                #         cv.circle(vis, (x, y), 2, (255, 255, 255))

            #self.rect_sel.draw(img_c)
            cv.imshow('Corr', img_c)
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == 27:
                break

# RealSense L or R App
class AppRS:
    def __init__(self, video_src = 'iid', paused = False):
        #self.cap = video.create_capture(video_src)
        self.cap        = RealSense(video_src)
        #self.cap        = RealSense('ggd')
        _, self.frame   = self.cap.read()
        frame_gray      = self.get_frame_gray(0)
        vis             = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR) 
        cv.imshow('frame left', vis)
        cv.imshow('frame right', vis)
        self.rect_sel   = RectSelector('frame left', self.onrect)
        self.rect_left  = []
        self.rect_right = []
        self.paused     = paused
        self.update_rate= 0
        self.corr       = CORR()

    def get_frame_gray(self, frame_type = 0):
        "extracts gray frame"
        if frame_type == 0:
            frame_gray = self.frame[:,:,0]
        elif frame_type == 1:
            frame_gray = self.frame[:,:,1]            
        else:
            frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        return frame_gray   

    def get_bbox(self, rect):
        "transform rect to bbox"
        x1, y1, x2, y2  = rect
        w, h            = map(cv.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1          = (x1+x2-w)//2, (y1+y2-h)//2
        x, y            = x1+0.5*(w-1), y1+0.5*(h-1) 
        return x,y,w,h

    def get_frame_roi(self, rect = [0,0,10,10], frame_type = 0):
        "extract rectangles"
        x,y,w,h         = self.get_bbox(rect)
        frame_gray      = self.get_frame_gray(frame_type)
        imgL            = cv.getRectSubPix(frame_gray, (w, h), (x, y))   
        return imgL

    def onrect(self, rect):
        #frame_gray      = self.frame #cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        #frame_gray      = self.get_frame_gray(0)
        self.rect_left.append(rect)  
        #rect[0]          += 10  # right image offset
        self.rect_right.append(rect)   

    def draw_rect(self, vis, rect):
        "rect on the image"
        x,y,w,h         = self.get_bbox(rect)
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        vis            = cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        #draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)
        return vis        

    def run(self):
        "main loop"
        imgC = None
        while True:
            if not self.paused:
                ret, frame_c = self.cap.read()
                if not ret:
                    break
                self.frame  = frame_c #[:,:,0]
           

            for k in range(len(self.rect_left)):
                imgL        = self.get_frame_roi(self.rect_left[k],  0)
                imgR        = self.get_frame_roi(self.rect_right[k], 1)
                imgC        = self.corr.correlate(imgL, imgR)
            
            vis_l = cv.cvtColor(self.frame[:,:,0], cv.COLOR_GRAY2BGR) 
            for rect in self.rect_left:
                vis_l = self.draw_rect(vis_l, rect)

            vis_r = cv.cvtColor(self.frame[:,:,1], cv.COLOR_GRAY2BGR) 
            for rect in self.rect_right:
                vis_r = self.draw_rect(vis_r, rect)         

            # draw during mouse move
            self.rect_sel.draw(vis_l)

            if imgC is not None:
                imgCB = np.uint8(imgC/imgC.max()*255)
                cv.imshow('Corr', imgCB)


            cv.imshow('frame left',  vis_l)
            cv.imshow('frame right', vis_r)
            ch = cv.waitKey(10)
            if ch == 27:
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.rect_left.pop()
                self.rect_right.pop()
            if ch == ord('u'):  
                #self.update_rate = 0.1 if self.update_rate < 0.001 else 0    
                for k in range(len(self.rect_left)):
                    rect = self.rect_left[k]
                    self.rect_left[k] = (rect[0]-1,rect[1],rect[2],rect[3])

        cv.destroyAllWindows()

# -------------------------- 
if __name__ == '__main__':
    #print(__doc__)

    #  #unittest.main()
    # suite = unittest.TestSuite()

    # #suite.addTest(TestCORR("test_1d"))
    # #suite.addTest(TestCORR("test_corr")) # ok
    # suite.addTest(TestCORR("test_corr_interpolated")) # ok

    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    AppRS('iid').run()



    