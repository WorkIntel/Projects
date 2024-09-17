
"""
Created on Sep 04 16:53:13 2019

2D STFT transform for correlation of a single pattern

Environment : 
    Pose6D

Installation:

Usage:


-----------------------------
 Ver    Date     Who    Descr
-----------------------------
0201   16.09.24 UD     Uses MOSSE correlation to match a single pattern
0103   14.09.24 UD     Correlation improvements with mean substraction.
0102   08.09.24 UD     Created as class.
0101   21.08.24 UD     Started. 
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
from common import log

sys.path.append(r'..\MonoDepth\src')
from peak_fit_2d import peak_fit_2d

sys.path.append(r'C:\Users\udubin\Documents\Code\opencv-4x\samples\python')
from common import draw_str, RectSelector

# ----------------------
#%% Helper function
def max_location(a):
    "position in 2d array"
    return unravel_index(a.argmax(), a.shape)


def rnd_warp(a):
    h, w        = a.shape[:2]
    T           = np.zeros((2, 3))
    coef        = 0.2
    ang         = (np.random.rand()-0.5)*coef
    c, s        = np.cos(ang), np.sin(ang)
    T[:2, :2]   = [[c,-s], [s, c]]
    T[:2, :2]  += (np.random.rand(2, 2) - 0.5)*coef
    c           = (w/2, h/2)
    T[:,2]      = c - np.dot(T[:2, :2], c)
    return cv.warpAffine(a, T, (w, h), borderMode = cv.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi + eps)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

# ----------------------
#%% Generate image and template
class DataGenerator:
    "class to create images and correspondance points"
    def __init__(self):

        self.frame_size = (640,480)
        self.imgD        = None
        self.imgL        = None
        self.imgR        = None

        self.roiR        = None

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

    def init_image(self, img_type = 1, window_size = 16):
        # create some images for test
        w,h             = self.frame_size   
        #window_size     = 16     

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
            self.imgR = np.roll(self.imgL, np.array([0, 0]), axis=(0, 1))
            offset    = [128,256]
            self.roiR = [offset[0], offset[1], offset[0]+window_size,offset[1]+window_size]             

        elif img_type == 4:
            self.imgD = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_029.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_031.png", cv.IMREAD_GRAYSCALE)
            self.imgR = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_030.png", cv.IMREAD_GRAYSCALE)  
            offset    = [128,128]
            self.roiR = [offset[0], offset[1], offset[0]+window_size,offset[1]+window_size]                        

        elif img_type == 5:
            self.imgD = cv.pyrDown(cv.imread(r"C:\Data\Corr\d2_Depth.png", cv.IMREAD_GRAYSCALE))
            self.imgL = cv.pyrDown(cv.imread(r"C:\Data\Corr\r3_Infrared.png", cv.IMREAD_GRAYSCALE))
            self.imgR = cv.pyrDown(cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE) )     
            self.imgR = np.roll(self.imgR, np.array([0, 0]), axis=(0, 1))
            offset    = [256,128]
            self.roiR = [offset[0], offset[1], offset[0]+window_size,offset[1]+window_size]                      

        elif img_type == 6: # same image with shift
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)
            self.imgR = np.roll(self.imgL, np.array([-4, -4]), axis=(0, 1))
            offset    = 32
            self.roiR = [offset, offset, offset+window_size,offset+window_size] 

        elif img_type == 7: # same image with shift - sub region
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)[100:328, 300:528]
            self.imgR = np.roll(self.imgL, np.array([-8, -8]), axis=(0, 1))
            offset    = 32
            self.roiR = [offset, offset, offset+window_size,offset+window_size]  

        elif img_type == 8: # same image but small
            self.imgD = cv.pyrDown(cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE))
            self.imgL = cv.pyrDown(cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE))
            self.imgR = np.roll(self.imgL, np.array([0, 0]), axis=(0, 1))
            offset    = [40,128]
            self.roiR = [offset[0],offset[1],offset[0]+window_size,offset[1]+window_size]                    

        # elif img_type == 11:  # Test patterns for correlation
        #     image_patch = np.ones((16, 16))
        #     image_patch[4:13, 4:13] = 0
        #     image       = np.tile(image_patch, (8, 8))
        #     image1      = image * 128 + 10 + np.random.randn(*image.shape) * 8
        #     shift       = np.array([2, 2]) * 1
        #     image2      = np.roll(image1, shift, axis=(0, 1))
        #     self.imgL, self.imgR = image1, image2

        elif img_type == 12:  # Test pattern against image - in sync
            image1      = np.random.rand(128, 128) * 60 + 40
            image2      = image1.copy()
            offset      = 32
            rect2       = [offset, offset, offset+window_size,offset+window_size]            
            shift       = np.array([2, 2]) *0
            image1      = np.roll(image1, shift, axis=(0, 1))  
            self.imgL, self.imgR, self.roiR = image1, image2, rect2

        elif img_type == 13:  # Test pattern against image - half win shift
            image1      = np.random.rand(128, 128) * 60 + 60
            image2      = image1.copy()
            offset      = 40
            rect2       = [offset, offset, offset+window_size,offset+window_size]            
            shift       = np.array([2, 2]) * 0
            image1      = np.roll(image1, shift, axis=(0, 1))  
            self.imgL, self.imgR, self.roiR = image1, image2, rect2

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
                      
        #self.frame_size = self.imgL.shape[::-1] # w,h 
        self.imgL, self.imgR = np.uint8(self.imgL), np.uint8(self.imgR)

        #self.img = self.add_noise(self.img, 0)
        #self.img = cv.resize(self.img , dsize = self.frame_size)   
        #imgL = cv.pyrDown(self.imgL)  # downscale images for faster processing
        #imgR = cv.pyrDown(self.imgR)
        #self.roiR       = self.init_roi(roi_type, window_size)
              
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
      
    def init_roi(self, test_type = 0, window_size = 16):
        "load the roi case"
        w,h     = self.frame_size
        w2, h2  = w>>1, h>>1
        roi     = [0,0,w,h]
        if test_type == 1:
            roi = [w2-3,h2-3,w2+3,h2+3] # xlu, ylu, xrb, yrb
        elif test_type == 2:
            roi = [300, 220, 300 + window_size, 220 + window_size] # xlu, ylu, xrb, yrb
        elif test_type == 3:
            roi = [280,200,360,280] # xlu, ylu, xrb, yrb            
        elif test_type == 4:
            roi = [220,140,420,340] # xlu, ylu, xrb, yrb      
        elif test_type == 5:
            roi = [200,120,440,360] # xlu, ylu, xrb, yrb     
   
        return roi   
    
    def show_images(self):
        "show images left and right"
        image_out   = np.concatenate((self.imgL, self.imgR), axis = 1) 
        image_out   = np.uint8(image_out)
        while image_out.shape[1] > 1999:
            image_out = cv.impyrDown(image_out)

        cv.imshow('Input Images', image_out)
        ch = cv.waitKey(1) & 0xff

# ----------------------
#%% Correlator
class STFT2D:
    def __init__(self, frame, rect):
        x1, y1, x2, y2  = rect
        w, h            = map(cv.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1          = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w), y1+0.5*(h) #x1+0.5*(w-1), y1+0.5*(h-1)
        self.size       = w, h
        #img             = cv.getRectSubPix(frame, (w, h), (x, y))
        img             = frame[y1:y2,x1:x2]
        self.win        = cv.createHanningWindow((w, h), cv.CV_32F)
        self.last_img   = img
        g               = np.zeros((h, w), np.float32)
        g[h//2, w//2]   = 1
        g               = cv.GaussianBlur(g, (-1, -1), 9.0)
        g              /= g.max()
        #self.G          = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)
        self.G          = np.fft.fftshift(1-g)

        # self.H1         = np.zeros_like(self.G)
        # self.H2         = np.zeros_like(self.G)
        # for _i in xrange(16): #128):
        #     imgr        = rnd_warp(img)
        #     a           = self.preprocess(imgr)
        #     #a          = self.preprocess(img)
        #     A           = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
        #     self.H1    += A #cv.mulSpectrums(self.G, A, 0, conjB=True)
        #     self.H2    += cv.mulSpectrums(     A, A, 0, conjB=True)
        #     #self.H  = A

        self.init_kernel(img)
        self.update_kernel()
        #self.update(frame)

    def preprocess(self, img):
        img         = np.log(np.float32(img)+1.0)
        img         = img-img.mean()  # important line
        #img         = img/ (img.std()+eps)  # important line
        return img*self.win   

    def subpixel_peak_offsets(self,peak_vals):
        # deals with border
        if peak_vals.shape[0] != 5 or peak_vals.shape[1]!=5:
            return 0,0
        
        xg = np.array([[-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.]])
        yg = np.array([[-2., -2., -2., -2., -2.],       [-1., -1., -1., -1., -1.],       [ 0.,  0.,  0.,  0.,  0.],       [ 1.,  1.,  1.,  1.,  1.],       [ 2.,  2.,  2.,  2.,  2.]])
        w  = peak_vals.sum()
        dx = np.multiply(peak_vals,xg).sum()/w
        dy = np.multiply(peak_vals,yg).sum()/w
        return dx, dy

    def init_kernel(self, img):
        "instead of init in the __init__ function"
        h, w            = img.shape
        self.H1         = np.zeros((h,w,2),dtype = np.float32 ) #np.zeros_like(self.G)
        self.H2         = np.zeros_like(self.H1)
        for _i in range(128): #128):
            imgr        = rnd_warp(img)
            a           = self.preprocess(imgr)
            A           = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
            #A[0,0,:]    = eps # no DC
            A[:,:,0],A[:,:,1] = A[:,:,0]*self.G,A[:,:,1]*self.G # no dc
            self.H1    += A #cv.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2    += cv.mulSpectrums(A, A, 0, conjB=True)

    def update_kernel(self):
        self.H          = divSpec(self.H1, self.H2)
        #self.H[...,1] *= -1    
        
    def correlate(self, img):
        A           = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        # A[:,:,0],A[:,:,1] = A[:,:,0]*self.G,A[:,:,1]*self.G # no dc
        # A_norm      = 1/(cv.magnitude(A[:,:,0],A[:,:,1])+eps)
        # A[:,:,0],A[:,:,1] = A[:,:,0]*A_norm,A[:,:,1]*A_norm

        C           = cv.mulSpectrums(A, self.H, 0, conjB=True)
        resp        = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        resp        = np.fft.fftshift(resp) # Shift the zero-frequency component to the center

        h, w        = resp.shape
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)
        side_resp   = resp.copy()
        #cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr         = (mval-smean) / (sstd+eps)

        # sub pixel
        search_size = 3
        z           = resp[my-search_size:my+search_size, mx-search_size:mx+search_size]
        xp, yp      = peak_fit_2d(z)
        xp, yp      = xp - search_size, yp - search_size     
        mx, my      = mx + xp, my + yp
        #print(f"{my:.2f},{mx:.2f}: {yp:.2f},{xp:.2f}")
        #print(f"{my:.2f},{mx:.2f}")
        #time.sleep(0.5)

        # # UD  subpixel
        # respc       = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_COMPLEX_OUTPUT)
        # respc       = np.fft.fftshift(respc)
        # resp        = cv.magnitude(respc[:,:,0],respc[:,:,1])   
        # cval        = respc[my,mx,:].squeeze()
        # angl        = np.arctan2(cval[1],cval[0])*180/np.pi
        # print(f"{my},{mx}: {cval[0]:.2f},{cval[1]:.2f} : {angl:.2f}")
        # side_resp = respc.copy()
        #cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)        

        return resp, (mx-w//2, my-h//2), psr    
    
    def update(self, frame, rate = 0.0): #125):
        (x, y), (w, h)      = self.pos, self.size
        #img = cv.getRectSubPix(frame, (w, h), (x, y))
        x1, y1              = np.int32(x - w//2), np.int32(y - h//2)
        img                 = frame[y1:y1+h,x1:x1+w]

        self.last_img       = img
        img                 = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good           = self.psr > 8.0
        if not self.good:
            return

        self.pos            = x+dx, y+dy
        #self.last_img = img = cv.getRectSubPix(frame, (w, h), self.pos)
        x1, y1              = np.round(self.pos[0] - w//2).astype(np.int32), np.round(self.pos[1] - h//2).astype(np.int32)
        self.last_img = img = frame[y1:y1+h,x1:x1+w]
        img                 = self.preprocess(img)
        A                   = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        #H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
        H1                  = A
        H2                  = cv.mulSpectrums(A, A, 0, conjB=True)
        self.H1             = self.H1 * (1.0-rate) + H1 * rate
        self.H2             = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()
        #self.H  = A    

    def block_process(self, big_array: np.array):
        """ 
        Prforms block processing of the big array using small arrray and function provided
        Similar to Matlab block processing function

        number of rows and columns in big array must be interger multiple of the small array
        fun - should accept two arrays of the asme size and return the same size array
        
        """
        small_array = self.H

        br,bc       = big_array.shape
        sr,sc       = small_array.shape[:2]

        row_num, col_num = br//sr, bc//sc
        if not isinstance(row_num, int) or not isinstance(col_num, int):
            raise ValueError("number of rows and columns in big array must be interger multiple of the small array")
        
        a1      = big_array.reshape(row_num, sr, col_num, sc).transpose(0, 2, 1, 3)
        res1    = np.zeros(a1.shape)

        for r in range(row_num):
            for c in range(col_num):
                img              = a1[r,c,:,:]
                imgp             = self.preprocess(img)
                A                = cv.dft(imgp, flags=cv.DFT_COMPLEX_OUTPUT)    
                #A[:,:,0],A[:,:,1] = A[:,:,0]*self.G,A[:,:,1]*self.G # no dc    
                A_norm          = 1/(cv.magnitude(A[:,:,0],A[:,:,1])+eps)
                A[:,:,0],A[:,:,1] = A[:,:,0]*A_norm,A[:,:,1]*A_norm
                C                = cv.mulSpectrums(A, small_array, 0, conjB=True) #/cv.norm(A)
                resp             = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
                res1[r,c,:,:]    = np.fft.fftshift(resp)
                #res1[r,c,:,:]    = np.fft.fftshift(resp[:,:,0] +1j*resp[:,:,1])

        res     = res1.transpose(0, 2, 1, 3).reshape(big_array.shape)
        return res

    def correlate_frame(self, frame):
        "work on the entire frame"
        # size of the patch
        window_size      = self.size[0]

        # Get image dimensions and number of color channels
        n_rows, n_cols   = frame.shape

        # Ensure image size is compatible with window size
        max_row_win     = n_rows - window_size // 2
        max_col_win     = n_cols - window_size // 2
        if max_row_win < window_size or max_col_win < window_size:
            raise ValueError("Image size is less than the size of the window")

        # Calculate number of windows in each direction
        row_win_num     = max_row_win // window_size
        col_win_num     = max_col_win // window_size

        # Define active pixel region
        active_rows     = row_win_num * window_size
        active_cols     = col_win_num * window_size

        # Define translations
        translations = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * (window_size // 2)   

        # Initialize correlation output
        corr_result  = np.zeros((n_rows, n_cols), dtype=np.float32)

        # Loop through translations and perform STFT
        for i, translation in enumerate(translations):

            image_patch    = frame[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols] #
            corr_patch     = self.block_process(image_patch)
            corr_result[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols] += corr_patch

            # debug
            peak_xy    = max_location(np.abs(corr_patch))
            plt.figure(20 + i)
            plt.imshow(np.abs(corr_patch), cmap='gray')
            plt.title('Pose y-%s, x-%s peak at %s' %(str(translation[0]),str(translation[1]),str(peak_xy)))
            plt.colorbar() #orientation='horizontal')
            plt.show()
                     
        return np.real(corr_result)
  
    @property
    def state_vis(self):
        f               = cv.idft(self.H, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT )
        h, w            = f.shape
        f               = np.roll(f, -h//2, 0)
        f               = np.roll(f, -w//2, 1)
        kernel          = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp            = self.last_resp
        resp            = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis             = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def show_corr_image(self, correlation_image, fig_num = 31):
        "shows the correlation image"
        peak_xy           = max_location(np.abs(correlation_image))
        peak_val          = np.abs(correlation_image.max())
        txts              = 'Max val : %s, position : %s' %(str(peak_val), str(peak_xy))
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

# ----------------------
#%% Peak Tracker
class STFT2D_ORIG:
    def __init__(self, frame, rect):
        x1, y1, x2, y2  = rect
        w, h            = map(cv.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1          = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size       = w, h
        img             = cv.getRectSubPix(frame, (w, h), (x, y))
        #img             = frame[y1:y2,x1:x2]
        self.win        = cv.createHanningWindow((w, h), cv.CV_32F)
        self.last_img   = img
        g               = np.zeros((h, w), np.float32)
        g[h//2, w//2]   = 1
        g               = cv.GaussianBlur(g, (-1, -1), 2.0)
        g              /= g.max()
        self.G          = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)

        self.init_kernel(img)
        self.update_kernel()
        self.update(frame)

    def preprocess(self, img):
        img         = np.log(np.float32(img)+1.0)
        img         = img-img.mean()  # important line
        img         = img/ (img.std()+eps)  # important line
        return img*self.win   

    def subpixel_peak_offsets(self,peak_vals):
        # deals with border
        if peak_vals.shape[0] != 5 or peak_vals.shape[1]!=5:
            return 0,0
        
        xg = np.array([[-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.],       [-2., -1.,  0.,  1.,  2.]])
        yg = np.array([[-2., -2., -2., -2., -2.],       [-1., -1., -1., -1., -1.],       [ 0.,  0.,  0.,  0.,  0.],       [ 1.,  1.,  1.,  1.,  1.],       [ 2.,  2.,  2.,  2.,  2.]])
        w  = peak_vals.sum()
        dx = np.multiply(peak_vals,xg).sum()/w
        dy = np.multiply(peak_vals,yg).sum()/w
        return dx, dy

    def init_kernel(self, img):
        "instead of init in the __init__ function"
        self.H1         = np.zeros_like(self.G)
        self.H2         = np.zeros_like(self.G)
        for _i in range(128): #128):
            imgr        = rnd_warp(img)
            a           = self.preprocess(imgr)
            A           = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
            self.H1    += cv.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2    += cv.mulSpectrums(A, A, 0, conjB=True)

    def update_kernel(self):
        self.H          = divSpec(self.H1, self.H2)
        self.H[...,1]  *= -1    
        
    def correlate(self, img):
        A           = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        C           = cv.mulSpectrums(A, self.H, 0, conjB=True)
        resp        = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        #resp       = np.fft.fftshift(resp) # Shift the zero-frequency component to the center

        h, w        = resp.shape
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)
        side_resp   = resp.copy()
        #cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr         = (mval-smean) / (sstd+eps)

        # sub pixel
        # search_size = 3
        # z           = resp[my-search_size:my+search_size, mx-search_size:mx+search_size]
        # xp, yp      = peak_fit_2d(z)
        # xp, yp      = xp - search_size, yp - search_size     
        # mx, my      = mx + xp, my + yp
        #print(f"{my:.2f},{mx:.2f}: {yp:.2f},{xp:.2f}")
        #print(f"{my:.2f},{mx:.2f}")
        #time.sleep(0.5)

        # # UD  subpixel
        # respc       = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_COMPLEX_OUTPUT)
        # respc       = np.fft.fftshift(respc)
        # resp        = cv.magnitude(respc[:,:,0],respc[:,:,1])   
        # cval        = respc[my,mx,:].squeeze()
        # angl        = np.arctan2(cval[1],cval[0])*180/np.pi
        # print(f"{my},{mx}: {cval[0]:.2f},{cval[1]:.2f} : {angl:.2f}")
        # side_resp = respc.copy()
        #cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)        

        return resp, (mx-w//2, my-h//2), psr    
    
    def update(self, frame, rate = 0.0): #125):
        (x, y), (w, h)      = self.pos, self.size
        img                 = cv.getRectSubPix(frame, (w, h), (x, y))
        self.last_img       = img
        img                 = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good           = self.psr > 8.0
        if not self.good:
            return

        self.pos            = x+dx, y+dy
        self.last_img = img = cv.getRectSubPix(frame, (w, h), self.pos)
        img                 = self.preprocess(img)
        A                   = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        H1                  = cv.mulSpectrums(self.G, A, 0, conjB=True)
        H2                  = cv.mulSpectrums(     A, A, 0, conjB=True)
        self.H1             = self.H1 * (1.0-rate) + H1 * rate
        self.H2             = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()


    def block_process(self, big_array: np.array):
        """ 
        Prforms block processing of the big array using small arrray and function provided
        Similar to Matlab block processing function

        number of rows and columns in big array must be interger multiple of the small array
        fun - should accept two arrays of the asme size and return the same size array
        
        """
        small_array = self.H

        br,bc       = big_array.shape
        sr,sc       = small_array.shape[:2]

        row_num, col_num = br//sr, bc//sc
        if not isinstance(row_num, int) or not isinstance(col_num, int):
            raise ValueError("number of rows and columns in big array must be interger multiple of the small array")
        
        a1      = big_array.reshape(row_num, sr, col_num, sc).transpose(0, 2, 1, 3)
        res1    = np.zeros(a1.shape)

        for r in range(row_num):
            for c in range(col_num):
                img              = a1[r,c,:,:]
                imgp             = self.preprocess(img)
                A                = cv.dft(imgp, flags=cv.DFT_COMPLEX_OUTPUT)        
                C                = cv.mulSpectrums(A, small_array, 0, conjB=True) #/cv.norm(A)
                resp             = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
                res1[r,c,:,:]    = resp #np.fft.fftshift(resp)
                #res1[r,c,:,:]    = np.fft.fftshift(resp[:,:,0] +1j*resp[:,:,1])

        res     = res1.transpose(0, 2, 1, 3).reshape(big_array.shape)
        return res

    def correlate_frame(self, frame):
        "work on the entire frame"
        # size of the patch
        window_size      = self.size[0]

        # Get image dimensions and number of color channels
        n_rows, n_cols   = frame.shape

        # Ensure image size is compatible with window size
        max_row_win     = n_rows - window_size // 2
        max_col_win     = n_cols - window_size // 2
        if max_row_win < window_size or max_col_win < window_size:
            raise ValueError("Image size is less than the size of the window")

        # Calculate number of windows in each direction
        row_win_num     = max_row_win // window_size
        col_win_num     = max_col_win // window_size

        # Define active pixel region
        active_rows     = row_win_num * window_size
        active_cols     = col_win_num * window_size

        # Define translations
        translations = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * (window_size // 2)   

        # Initialize correlation output
        corr_result  = np.zeros((n_rows, n_cols), dtype=np.float32)

        # Loop through translations and perform STFT
        for i, translation in enumerate(translations):

            image_patch    = frame[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols] #
            corr_patch     = self.block_process(image_patch)
            corr_result[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols] += corr_patch

            # debug
            peak_xy    = max_location(corr_patch)
            plt.figure(20 + i)
            plt.imshow(np.abs(corr_patch), cmap='gray')
            plt.title('Pose y-%s, x-%s peak at %s' %(str(translation[0]),str(translation[1]),str(peak_xy)))
            plt.colorbar() #orientation='horizontal')
            plt.show()
                     
        return np.real(corr_result)
  
    @property
    def state_vis(self):
        f               = cv.idft(self.H, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT )
        h, w            = f.shape
        f               = np.roll(f, -h//2, 0)
        f               = np.roll(f, -w//2, 1)
        kernel          = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp            = self.last_resp
        resp            = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis             = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def show_corr_image(self, correlation_image, fig_num = 31):
        "shows the correlation image"
        peak_xy           = max_location(np.abs(correlation_image))
        peak_val          = np.abs(correlation_image.max())
        txts              = 'Max val : %s, position : %s' %(str(peak_val), str(peak_xy))
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


# ----------------------
#%% Tests
class TestSTFT2D(unittest.TestCase):

    def test_stft2d_corr(self):
        "correlator"
        w_size  = 32
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 12, window_size = w_size)
        d.show_images()

        c       = STFT2D(d.imgR, d.roiR)
        img_c   = c.correlate_frame(d.imgL)
        c.show_corr_image(img_c)
        self.assertTrue(isOk)   

    def test_stft2d_corr_shift(self):
        "correlator with shifted pattern"
        w_size  = 32
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 13, window_size = w_size)
        d.show_images()

        c       = STFT2D(d.imgR, d.roiR)
        img_c   = c.correlate_frame(d.imgL)
        c.show_corr_image(img_c)
        self.assertTrue(isOk)    

    def test_single_image_with_itself(self):
        "correlator with single image"
        w_size  = 32
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 8, window_size = w_size) # 6,7,8-ok
        d.show_images()

        c       = STFT2D(d.imgR, d.roiR)
        img_c   = c.correlate_frame(d.imgL)
        c.show_corr_image(img_c)
        self.assertTrue(isOk)                     

    def test_two_images(self):
        "correlator between 2 images"
        w_size  = 32
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 3, window_size = w_size) # 4-?,5-ok
        d.show_images()

        c       = STFT2D(d.imgR, d.roiR)
        img_c   = c.correlate_frame(d.imgL)
        c.show_corr_image(img_c)
        self.assertTrue(isOk)  

    def test_original_corr(self):
        "correlator with orig implementation"
        w_size  = 32
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 13, window_size = w_size) # 12,13 - ok
        d.show_images()

        c       = STFT2D_ORIG(d.imgR, d.roiR)
        img_c   = c.correlate_frame(d.imgL)
        c.show_corr_image(img_c)
        self.assertTrue(isOk)    

    def test_original_corr_single_image_with_itself(self):
        "correlator with single image"
        w_size  = 32
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 7, window_size = w_size) # 6,7-ok
        d.show_images()

        c       = STFT2D_ORIG(d.imgR, d.roiR)
        img_c   = c.correlate_frame(d.imgL)
        c.show_corr_image(img_c)
        self.assertTrue(isOk)               

# ----------------------
#%% App
class App:
    def __init__(self):
        self.cap   = RealSense()
        self.cap.change_mode('iid')

        self.corr = STFT2D()

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
            img_c   = self.corr.test_stft2d_corr(self.frame[:,:,0], self.frame[:,:,1], window_size = 16)
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

      
# -------------------------- 
if __name__ == '__main__':
    #print(__doc__)

     #unittest.main()
    suite = unittest.TestSuite()

    #suite.addTest(TestSTFT2D("test_stft2d_corr")) # ok
    #suite.addTest(TestSTFT2D("test_stft2d_corr_shift")) # ok
    #suite.addTest(TestSTFT2D("test_single_image_with_itself")) # ok
    suite.addTest(TestSTFT2D("test_two_images"))

    #suite.addTest(TestSTFT2D("test_original_corr")) # ok
    #suite.addTest(TestSTFT2D("test_original_corr_single_image_with_itself")) # nok
 


    runner = unittest.TextTestRunner()
    runner.run(suite)



    