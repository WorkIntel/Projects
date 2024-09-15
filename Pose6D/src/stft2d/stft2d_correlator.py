
"""
Created on Sep 04 16:53:13 2019

2D STFT transform for correlation

Environment : 

Installation:

Usage:


-----------------------------
 Ver    Date     Who    Descr
-----------------------------
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

# ----------------------
#%% Helper function
def max_location(a):
    "position in 2d array"
    return unravel_index(a.argmax(), a.shape)

def multiply(a, b):
  """
  multiplies two arrays.
  """
  c = a * b
  return c

def fft_multiply(a, b):
  """
  performs FFT2 and then multiplies two arrays.
  """
  c = fft2(a) * b
  return c

def ifft_multiply(a, b):
  """
  performs IFFT2 and then mult two arrays.
  """
  c = ifft2(a) * b
  return c

def subtract_mean(a, d):
  """
  removes mean from 3 and 4 dim.
  """
  b = a.mean(axis = (2,3), keepdims = True)
  c = a - b
  return c

def block_process(big_array: np.array, small_array: np.array, fun):
    """ 
    Prforms block processing of the big array using small arrray and function provided
    Similar to Matlab block processing function

    number of rows and columns in big array must be interger multiple of the small array
    fun - should accept two arrays of the asme size and return the same size array
    
    """
    br,bc = big_array.shape
    sr,sc = small_array.shape

    row_num, col_num = br//sr, bc//sc
    if not isinstance(row_num, int) or not isinstance(col_num, int):
        raise ValueError("number of rows and columns in big array must be interger multiple of the small array")
    
    a1      = big_array.reshape(row_num, sr, col_num, sc).transpose(0, 2, 1, 3)
    res1    = fun(a1 , small_array)
    res     = res1.transpose(0, 2, 1, 3).reshape(big_array.shape)
    return res

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

        elif img_type == 5:
            self.imgD = cv.pyrDown(cv.imread(r"C:\Data\Corr\d2_Depth.png", cv.IMREAD_GRAYSCALE))
            self.imgL = cv.pyrDown(cv.imread(r"C:\Data\Corr\l2_Infrared.png", cv.IMREAD_GRAYSCALE))
            self.imgR = cv.pyrDown(cv.imread(r"C:\Data\Corr\r2_Infrared.png", cv.IMREAD_GRAYSCALE) )             

        elif img_type == 6: # same image with shift
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)
            self.imgR = np.roll(self.imgL, np.array([-4, -4]), axis=(0, 1))

        elif img_type == 7: # same image with shift - sub region
            self.imgD = cv.imread(r"C:\Data\Corr\d3_Depth.png", cv.IMREAD_GRAYSCALE)
            self.imgL = cv.imread(r"C:\Data\Corr\l3_Infrared.png", cv.IMREAD_GRAYSCALE)[100:328, 300:528]
            self.imgR = np.roll(self.imgL, np.array([-4, -4]), axis=(0, 1))

        elif img_type == 11:  # Test patterns for correlation
            image_patch = np.ones((16, 16))
            image_patch[4:13, 4:13] = 0
            image       = np.tile(image_patch, (8, 8))
            image1      = image * 128 + 10 + np.random.randn(*image.shape) * 8
            shift       = np.array([2, 2]) * 1
            image2      = np.roll(image1, shift, axis=(0, 1))
            self.imgL, self.imgR = image1, image2

        elif img_type == 12:  # Test pattern against image - in sync
            image1      = np.random.rand(128, 128) * 60 + 0
            offset      = 32
            image_patch = image1[offset:offset+window_size,offset:offset+window_size]
            image2      = np.tile(image_patch, (8, 8))
            shift       = np.array([2, 2]) * 0
            image1      = np.roll(image1, shift, axis=(0, 1))  
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
    
    def show_images(self):
        "show images left and right"
        image_out   = np.concatenate((self.imgL, self.imgR), axis = 1) 
        image_out   = np.uint8(image_out)
        while image_out.shape[1] > 1999:
            image_out = cv.impyrDown(image_out)

        cv.imshow('Input Images', image_out)
        ch = cv.waitKey(1) & 0xff

# ----------------------
#%% Estimation
class STFT2D:
    """ 
    Uses 2D correlation for matching
    """

    def __init__(self):

        self.frame_size     = (640,480)

    def preprocess(self, img, corr_enabled = False):
        "image preprocessing - converts from uint8 to float using log function"
        img            = img.astype(np.float64)

        img            = np.log(img + 1.0) if corr_enabled else img
        #img            = (img-img.mean()) / (img.std()+1e-9)
        return img

    def stft2d(self, Im, window_size=16, corr_enabled=False, no_mean = False):
        """
        Performs 2D short-time Fourier transform (STFT) on a grayscale image.

        Args:
            Im: A 2D numpy array representing the input grayscale image.
            window_size: Size of the triangular window (default: 16).
            corr_enabled: Boolean flag enabling correlation adjustment (default: False).

        Returns:
            A 3D complex numpy array containing the STFT coefficients for 4 translations.

        Raises:
            ValueError: If window size is not even.
            ValueError: If image size is smaller than the window size.
        """

        # Check arguments
        if window_size % 2 != 0:
            raise ValueError("Window size must be even")

        if len(Im.shape) > 2:
            raise ValueError("Image must be 2D array")

        # Convert image to double and grayscale if necessary
        Im            = self.preprocess(Im, no_mean)

        # Get image dimensions and number of color channels
        n_rows, n_cols = Im.shape

        # Ensure image size is compatible with window size
        max_row_win = n_rows - window_size // 2
        max_col_win = n_cols - window_size // 2
        if max_row_win < window_size or max_col_win < window_size:
            raise ValueError("Image size is less than the size of the window")

        # Calculate number of windows in each direction
        row_win_num = max_row_win // window_size
        col_win_num = max_col_win // window_size

        # Define active pixel region
        active_rows = row_win_num * window_size
        active_cols = col_win_num * window_size

            # Define translations
        translations      = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * (window_size // 2)

        # Construct window mask
        window_mask = np.outer(triang(window_size), triang(window_size))
        #mask        = np.tile(window_mask, (row_win_num, col_win_num))

        # Construct frequency mask for correlation adjustment (if enabled)
        dc_mask     = np.ones((window_size, window_size))
        if corr_enabled :
            #center_idx = window_size // 2
            #dc_mask[center_idx-1:center_idx+1, center_idx-1:center_idx+1] = 0
            #dc_mask[0,0]             = 0
            #dc_mask[window_size-1,0] = 0
            #dc_mask[0,window_size-1] = 0
            #dc_mask[window_size-1,window_size-1] = 0
            translations             = translations * 0  # no offsets
        #    frequency_mask    = window_mask
        #else:
        frequency_mask      = dc_mask #fftshift(dc_mask) # np.tile(fftshift(dc_mask), (row_win_num, col_win_num))

        # Initialize STFT output
        stft              = np.zeros((n_rows, n_cols, 4), dtype=np.complex64)

        # Loop through translations and perform STFT
        for i, translation in enumerate(translations):
            #row_indices   = np.arange(active_rows) + translation[0]
            #col_indices   = np.arange(active_cols) + translation[1]

            image_patch   = Im[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols] #

            if no_mean:
                image_patch   = block_process(image_patch, window_mask, subtract_mean)

            image_patch   = block_process(image_patch, window_mask, multiply)
            fft_patch     = block_process(image_patch, frequency_mask, fft_multiply)

            #stft[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols, i] = fft_patch
            stft[0:active_rows, 0:active_cols, i] = fft_patch

        return stft
    
    def istft2d(self, ImF, window_size=16, corr_enabled=False):
        """
        Performs inverse 2D short-time Fourier transform (ISTFT) on a complex frequency 
        domain representation.

        Args:
            ImF: A 3D complex numpy array containing STFT coefficients for 4 translations.
            window_size: Size of the triangular window used in STFT (default: 16).
            corr_enabled: Boolean flag indicating if correlation adjustment was used in STFT (default: False).

        Returns:
            A 2D numpy array representing the reconstructed image.

        Raises:
            ValueError: If window size is not even.
            ValueError: If the input array does not have 4 dimensions.
            ValueError: If image size is smaller than the window size.
        """

        # Check arguments
        if window_size % 2 != 0:
            raise ValueError("Window size must be even")

        # Check input dimensions
        if len(ImF.shape) != 3:
            raise ValueError("Input array must have 4 dimensions")

        # Get image dimensions
        n_rows, n_cols, n_dims = ImF.shape
        if n_dims != 4:
            raise ValueError("Input array must have 4 dimensions")

        # Calculate number of windows in each direction
        row_win_num = (n_rows - window_size // 2) // window_size
        col_win_num = (n_cols - window_size // 2) // window_size

        # Define active pixel region
        active_rows = row_win_num * window_size
        active_cols = col_win_num * window_size

        # Check image size
        if active_rows < window_size or active_cols < window_size:
            raise ValueError("Image size is less than the size of the window")

        # Initialize reconstructed image
        image         = np.zeros((n_rows, n_cols), dtype=np.complex64)

        # freq mask
        freq_mask     = np.ones((window_size,window_size))
        #freq_mask[0,0]= 0

        # Construct window mask
        window_mask   = np.ones((window_size,window_size)) #np.outer(triang(window_size), triang(window_size))

        # Define translations
        translations = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * (window_size // 2)

        # For correlation
        #if corr_enabled:
        #  translations = translations * 0  # no offsets


        # Loop through translations and perform inverse STFT
        for i, translation in enumerate(translations):

            fft_patch       = ImF[0:active_rows, 0:active_cols, i]

            # Apply inverse FFT
            image_patch       = block_process(fft_patch, freq_mask, ifft_multiply)
            image_patch       = block_process(image_patch, window_mask, multiply)

            # # Handle correlation adjustment if enabled
            # if corr_enabled:
            #   image_patch *= np.ones((window_size, window_size))  # No correction needed

            # Overlap-add reconstructed patches
            image[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols] += image_patch

            # debug
            peak_xy    = max_location(np.abs(image_patch))
            plt.figure(20 + i)
            plt.imshow(np.abs(image_patch), cmap='gray')
            plt.title('Pose y-%s, x-%s peak at %s' %(str(translation[0]),str(translation[1]),str(peak_xy)))
            plt.colorbar() #orientation='horizontal')
            plt.show()

        # Handle real output if correlation adjustment was not used
        if not corr_enabled:
            image = np.real(image)

        return image
    
    def test_stft2d(self, window_size=32, corr_enabled=False, test_type=1, fig_num=2):
        """
        Tests the 2D STFT decomposition.

        Args:
            window_size: Size of the triangular window (default: 32).
            corr_enabled: Boolean flag enabling correlation adjustment (default: False).
            test_type: Integer specifying the test case (default: 2).
            fig_num: Figure number for visualization (default: 2).
        """

        # Select test case
        if test_type == 1:  # Test random image
            image = np.random.rand(256, 256) * 30 + 128
        elif test_type == 2:  # Test simple image
            # Assuming 'circuit' is a predefined image
            image = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_1004.png")
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        elif test_type == 3:  # Test different size
            image = np.random.rand(160, 192) * 10
            image[50:58, 50:58] = 180
        elif test_type == 11:  # Test patterns for correlation
            image_patch = np.ones((16, 16))
            image_patch[4:13, 4:13] = 0
            image = np.tile(image_patch, (8, 8))
            image = image * 128 + 10 + np.random.randn(*image.shape) * 8
        else:
            raise ValueError("Unknown TestType")

        # Perform STFT and ISTFT
        stft                  = self.stft2d(image, window_size, corr_enabled)
        reconstructed_image   = self.istft2d(stft, window_size, corr_enabled)

        # Compare ignoring borders
        border_mask           = np.zeros_like(image)
        border_mask[window_size+1:-window_size, window_size+1:-window_size] = 1
        indices               = np.nonzero(border_mask)
        error                 = np.std(image[indices] - reconstructed_image[indices])
        print("Error:", error)

        # Visualize
        plt.figure(fig_num + 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.colorbar() #orientation='horizontal')

        plt.figure(fig_num + 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title("Reconstructed Image")
        plt.colorbar() #orientation='horizontal')

        plt.show()

    def test_stft2d_corr(self, image1, image2, window_size=16, corr_enabled=True, fig_num=1):
        """
        Tests 2D STFT decomposition based on image correlation.

        Args:
            window_size: Size of the triangular window (default: 16).
            corr_enabled: Boolean flag enabling correlation adjustment (default: True).
            test_type: Integer specifying the test case (default: 6).
            fig_num: Figure number for visualization (default: 1).
        """

        # Check image sizes
        if image1.shape != image2.shape:
            raise ValueError("Images must be of the same size")

        # Perform STFT
        image1_stft       = self.stft2d(image1, window_size, corr_enabled = False, no_mean = True) #, corr_enabled = False)
        image2_stft       = self.stft2d(image2, window_size, corr_enabled = True, no_mean = True)

        # Correlate and perform ISTFT
        correlation_stft  = image1_stft * np.conj(image2_stft)
        #correlation_stft  = image1_stft * np.repeat(np.conj(image2_stft[:,:,0])[:,:,np.newaxis],4,axis=2)
        correlation_image = self.istft2d(correlation_stft, window_size, corr_enabled = True)

        return correlation_image
        
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
           
        #print(ptxt)

# ----------------------
#%% Tests
class TestSTFT2D(unittest.TestCase):

    def test_stft2d(self):
        "direct and inverse"
        p       = STFT2D()
        p.test_stft2d()
        self.assertFalse(p.frame_size is None)  

    def test_stft2d_corr(self):
        "correlator"
        w_size  = 16
        p       = STFT2D()
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 12, window_size = w_size)
        img_c   = p.test_stft2d_corr(d.imgL, d.imgR, window_size = w_size)
        p.show_corr_image(img_c)
        self.assertTrue(isOk)          

    def test_stft2d_corr_datagen(self):
        "correlator and data gen"
        w_size  = 16
        p       = STFT2D()
        d       = DataGenerator()
        d.init_image(img_type = 13, window_size = w_size)
        img_c   = p.test_stft2d_corr(d.imgL, d.imgR, window_size = w_size)
        p.show_corr_image(img_c)
        self.assertFalse(img_c is None) 

    def test_stft2d_two_images(self):
        "correlator of left and right random images"
        w_size  = 32
        p       = STFT2D()
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 21, window_size = w_size)
        img_c   = p.test_stft2d_corr(d.imgL, d.imgR, window_size = w_size)
        d.show_images()
        p.show_corr_image(img_c)
        self.assertTrue(isOk)  

    def test_stft2d_two_real_small_images(self):
        "correlator of left and right small size images"
        w_size  = 32
        p       = STFT2D()
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 7, window_size = w_size)
        img_c   = p.test_stft2d_corr(d.imgL, d.imgR, window_size = w_size, corr_enabled = True)
        d.show_images()
        p.show_corr_image(img_c)
        self.assertTrue(isOk)           

    def test_stft2d_two_real_images(self):
        "correlator of left and right images"
        w_size  = 32
        p       = STFT2D()
        d       = DataGenerator()
        isOk    = d.init_image(img_type = 4, window_size = w_size)
        img_c   = p.test_stft2d_corr(d.imgL, d.imgR, window_size = w_size, corr_enabled = True)
        d.show_images()
        p.show_corr_image(img_c)
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

    #suite.addTest(TestSTFT2D("test_stft2d")) # ok
    #suite.addTest(TestSTFT2D("test_stft2d_corr")) # ok
    #suite.addTest(TestSTFT2D("test_stft2d_corr_datagen")) # ok
    #suite.addTest(TestSTFT2D("test_stft2d_two_images")) # ok
    #suite.addTest(TestSTFT2D("test_stft2d_two_real_small_images")) # ok
    suite.addTest(TestSTFT2D("test_stft2d_two_real_images")) # 

    runner = unittest.TextTestRunner()
    runner.run(suite)



    