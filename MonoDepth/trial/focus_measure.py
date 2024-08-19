"""
function FM = focusmeasure(Image, Measure, WSize)
%This function measures the relative degree of focus of 
%an image. It may be invoked as:
%
%   FM = focusmeasure(Image, Method, WSize)
%
%Where 
%   Image,  is a DOUBLE Image and FM is a 
%           matrix the same size as Image with the 
%           computed focus measure for every pixel. 
%   WSize,  is the size of the neighborhood used to 
%           compute the focus value of every pixel. 
%           If WSize = 0, a single focus measure is 
%           computed for the whole image and, in this 
%           case, FM is a scalar. 
%   Method, is the focus measure algorithm as a string.
%           
% S. Pertuz
% Jan/2016

"""

import numpy as np
import cv2

from scipy.ndimage import convolve, generic_filter
from scipy.signal import convolve2d
from scipy.stats import entropy
# from skimage.feature import local_binary_pattern
# from skimage.filters.rank import entropy as entropy_filter
# from skimage.filters import laplace, sobel, gaussian
# from skimage.measure import shannon_entropy
# from skimage.util import img_as_ubyte

# Define the focus measure methods as functions
def acmoment(window):
    # This is a placeholder for the actual AcMomentum function
    pass

def tchebifocus(window):
    # This is a placeholder for the actual TchebiFocus function
    pass

def eigenfocus(window):
    # This is a placeholder for the actual EigenFocus function
    pass

def dct_ratio(window):
    # This is a placeholder for the actual DctRatio function
    pass

def re_ratio(window):
    # This is a placeholder for the actual ReRatio function
    pass


def image_contrast(img_patch):
    return np.sum(np.abs(img_patch.flatten() - img_patch[1, 1]))

def contrast_focus_measure(image, wsize):
    if wsize == 0:
        fm = image_contrast(image)
    else:
        fm = generic_filter(image, image_contrast, size=(3, 3))
        fm = np.mean(fm)
    return fm

def brenner_focus_measure(image):
    """
    Computes Brenner's focus measure.

    Args:
        image: The input image as a numpy array.

    Returns:
        The computed focus measure as a float.
    """

    M, N = image.shape
    DH = np.zeros((M, N))
    DV = np.zeros((M, N))

    DV[1:M-1, :] = image[2:, :] - image[:-2, :]
    DH[:, 1:N-1] = image[:, 2:] - image[:, :-2]

    FM = np.maximum(DH, DV)
    FM = FM**2
    FM = np.mean(FM)

    return FM   

def curvature_focus_measure(image, wsize):
    M1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    M2 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])

    P0 = convolve(image, M1, mode='reflect') / 6
    P1 = convolve(image, M1.T, mode='reflect') / 6
    P2 = 3 * convolve(image, M2, mode='reflect') / 10 - convolve(image, M2.T, mode='reflect') / 5
    P3 = -convolve(image, M2, mode='reflect') / 5 + 3 * convolve(image, M2, mode='reflect') / 10

    FM = np.abs(P0) + np.abs(P1) + np.abs(P2) + np.abs(P3)

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = generic_filter(FM, np.mean, size=(wsize, wsize))

    return fm 


def dctm_focus_measure(image, wsize):
    M = np.array([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
    FM = convolve(image, M, mode='reflect')

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = generic_filter(FM, np.mean, size=(wsize, wsize))

    return fm

def gder_focus_measure(image, wsize):
    N = wsize // 2
    sigma = N / 3

    x, y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    G = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma)
    Gx = -x * G / (sigma**2)
    Gx /= np.sum(Gx)
    Gy = -y * G / (sigma**2)
    Gy /= np.sum(Gy)

    Rx = convolve(image, Gx, mode='reflect')
    Ry = convolve(image, Gy, mode='reflect')
    FM = Rx**2 + Ry**2

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = generic_filter(FM, np.mean, size=(wsize, wsize))

    return fm

def helm_focus_measure(image, wsize):
    mean_filter = np.ones((wsize, wsize)) / (wsize * wsize)
    U = convolve(image, mean_filter, mode='reflect')
    R1 = U / image
    R1[image == 0] = 1
    index = U > image
    FM = 1 / R1
    FM[index] = R1[index]

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = convolve(FM, mean_filter, mode='reflect')

    return fm

def grat_focus_measure(image, wsize):
    """
    Computes the thresholded gradient focus measure.

    Args:
        image: The input image as a numpy array.
        wsize: The size of the neighborhood for focus measure calculation.

    Returns:
        The computed focus measure as a numpy array.
    """

    Ix = image.copy()
    Iy = image.copy()
    Iy[:-1, :] = np.diff(image, axis=0)
    Ix[:, :-1] = np.diff(image, axis=1)
    FM = np.maximum(np.abs(Ix), np.abs(Iy))

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = generic_filter(FM, np.mean, size=(wsize, wsize))

    return fm

def lape_focus_measure(image, wsize):
    """
    Computes the energy of the Laplacian focus measure.

    Args:
        image: The input image as a numpy array.
        wsize: The size of the neighborhood for focus measure calculation.

    Returns:
        The computed focus measure as a numpy array.
    """

    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    FM = convolve(image, laplacian, mode='reflect')**2

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = generic_filter(FM, np.mean, size=(wsize, wsize))

    return fm

def lapm_focus_measure(image, wsize):
    """
    Computes the modified Laplacian focus measure.

    Args:
        image: The input image as a numpy array.
        wsize: The size of the neighborhood for focus measure calculation.

    Returns:
        The computed focus measure as a numpy array.
    """

    M = np.array([-1, 2, -1])
    Lx = convolve(image, M, mode='reflect')
    Ly = convolve(image, M[np.newaxis, :], mode='reflect')
    FM = np.abs(Lx) + np.abs(Ly)

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = generic_filter(FM, np.mean, size=(wsize, wsize))

    return fm

def tengthird_focus_measure(image, wsize):
    """
    Computes the Tenengrad focus measure.

    Args:
        image: The input image as a numpy array.
        wsize: The size of the neighborhood for focus measure calculation.

    Returns:
        The computed focus measure as a numpy array.
    """

    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gx = convolve(image, Sx, mode='reflect')
    Gy = convolve(image, Sx.T, mode='reflect')
    FM = Gx**2 + Gy**2

    if wsize == 0:
        fm = np.mean(FM)
    else:
        fm = generic_filter(FM, np.mean, size=(wsize, wsize))

    return fm

def focus_measure(image, measure, wsize):
    if wsize == 0 or wsize is None:
        wsize = 15
    mean_kernel = np.ones((wsize, wsize)) / (wsize ** 2)
    
    # Apply the selected focus measure method
    if measure.upper() == 'ACMO':
        if wsize == 0:
            fm = acmoment(img_as_ubyte(image))
        else:
            fm = generic_filter(img_as_ubyte(image), acmoment, size=(wsize, wsize))
            fm = convolve(fm, mean_kernel, mode='nearest')
    
    # Add other cases for different focus measures
    if measure.upper() == 'CONT':
        fm = contrast_focus_measure(image, wsize)  
    
    elif measure.upper() == 'BREN': # Brenner's (Santos97)
        fm = brenner_focus_measure(image)

    elif measure.upper() == 'CURV': # Image Curvature (Helmli2001)
        fm = curvature_focus_measure(image,wsize)
    
    elif measure.upper() == 'DCTM': # % DCT Modified (Lee2008)
        fm = dctm_focus_measure(image, wsize)

    elif measure.upper() == 'GDER': # Gaussian derivative (Geusebroek2000)
        fm = gder_focus_measure(image, wsize)

    elif measure.upper() == 'HELM' : #%Helmli's mean method (Helmli2001)
        fm = helm_focus_measure(image, wsize)

    elif measure.upper() == 'GRAT' : # Thresholded gradient (Snatos97)
        fm = grat_focus_measure(image, 0)

    elif measure.upper() == 'LAPE' : # Energy of laplacian (Subbarao92a)
        fm = lape_focus_measure(image, wsize)        

    elif measure.upper() == 'LAPM' : # Modified Laplacian (Nayar89)
        fm = lapm_focus_measure(image, wsize)  

    elif measure.upper() == 'TENG' : # Tenengrad (Krotkov86)
        fm = tengthird_focus_measure(image, wsize)  
        

    else:
        raise ValueError(f'Unknown measure {measure}')
    
    return fm


if __name__ == '__main__':
    # Example usage:
    # Load an image using OpenCV
    path_image = r'C:\Users\udubin\Documents\Projects\MonoDepth\data\simu02\im_03.tif'
    image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float64)  # Convert to double

    # Call the focus measure function
    #fm = focus_measure(image, 'CONT', 15) # nok
    #fm = focus_measure(image, 'BREN', 15)
    fm = focus_measure(image, 'GRAT', 15)
