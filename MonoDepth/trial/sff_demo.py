""" 
clear, clc, close all
%% Load the variables of the focus sequence:
load imdata.mat

% This loads a structure with two fields: images, a 1x30 cell
% array where each cell is a string with the path of one
% frame of the focus sequence; and focus, a 1x30 vector with
% the focus position (in meters) corresponding to each frame
% of the focus sequence. This sequence was generated using:
% http://www.mathworks.com/matlabcentral/fileexchange/55095-defocus-simulation

%% Preview the image sequence:
showimages(imdata.images, imdata.focus, imdata.ROI);

%% Perform SFF using 3-point gaussian interpolation
% as originally proposed by [1] and compute
% reliability according to [2]:

[z, r] = sff(imdata.images, 'focus', imdata.focus);

%% Carve depthmap by removing pixels with R<20 dB:
zc = z;
zc(r<20) = nan;

%% Display the result:
close(gcf), figure
subplot(1,2,1), surf(z), shading flat, colormap copper
set(gca, 'zdir', 'reverse', 'xtick', [], 'ytick', [])
axis tight, grid off, box on
zlabel('pixel depth (mm)')
title('Full depthmap')

subplot(1,2,2), surf(zc), shading flat, colormap copper
set(gca, 'zdir', 'reverse', 'xtick', [], 'ytick', [])
axis tight, grid off, box on
zlabel('pixel depth (mm)')
title('Carved depthmap (R<20dB)')


% References:
% [1] S.K. Nayar and Y. Nakagawa, PAMI, 16(8):824-831, 1994. 
% [2] S. Pertuz, D. Puig, M. A. Garcia, Reliability measure for shape-from
% focus, IMAVIS, 31:725�734, 2013.

"""

import scipy.io as sio     # Load MATLAB files 
from scipy import ndimage, misc
#from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d  
import cv2
import numpy as np
from sff import compute_sff
import glob

# -------------------------
def load_matdata(fpath):
    mat_contents = sio.loadmat(fpath)
    print(type(mat_contents))
    print(mat_contents)
    img_stack = mat_contents['imdata'][0][0]

    return img_stack

def read_images(imlist, opts):
    """
    Reads images from a list of file paths, performs cropping and grayscale conversion if necessary.

    Args:
        imlist: A list of image file paths.
        opts: A dictionary containing options.

    Returns:
        A numpy array of shape (M, N, P) containing the images.
    """

    M, N = opts['size'][0], opts['size'][1]
    P = len(imlist)
    images = np.zeros((M, N, P), dtype=np.uint8)

    for p, img_path in enumerate(imlist):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        if opts['ROI'] is not None:
            x, y, w, h = opts['ROI']
            img = img[y:y+h, x:x+w]

        if opts['RGB']:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images[:, :, p] = img
        print(f"\rReading [{p+1:02d}/{P:02d}]", end="")

    print()
    return images

def read_show_images(imdata, focus=None, roi=None):
    """
    Displays an image sequence.

    Args:
        images: A list of image paths.
        focus: A list of focus positions (optional).
        roi: A 4-element tuple representing the region of interest (optional).
    """
    imgpath = imdata[0][0]
    focus   = imdata['focus'][0]
    M,N,P   = 256, 256, len(imgpath)
    images  = np.zeros((M, N, P), dtype=np.uint8)
    #roi     = imdata[2]
    for k, img_path in enumerate(imgpath):
        file_path   = img_path[0].replace('./','./data/')
        img         = cv2.imread(file_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if focus is not None:
            plt.text(20, 20, f"focus = {focus[k]:.3f} [m]", fontsize=14, color='green')

        if roi is not None:
            x, y, w, h = roi
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='blue', fill=False))


        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images[:, :, k] = img
        #images = np.stack((images,img[:,:,np.newaxis]), axis = 2)
        #print(f"\rReading [{k+1:02d}/{P:02d}]", end="")

        plt.show(block=False)
        plt.pause(0.02)

    images = np.array(images)
    return images, focus

def read_razy_images(fpath, focus=None, roi=None):
    """
    Displays an image sequence sampled by Razy.

    Args:
        images: A list of image paths.
        focus: A list of focus positions (optional).
        roi: A 4-element tuple representing the region of interest (optional).
    """
    imgpath = glob.glob(fpath)
    M,N,P   = 480, 640, len(imgpath)
    focus   = np.arange(P)*10

    images  = np.zeros((M, N, P), dtype=np.uint8)
    #roi     = imdata[2]
    for k, img_path in enumerate(imgpath):
        file_path   = img_path #[0].replace('./','./data/')
        img         = cv2.imread(file_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if focus is not None:
            plt.text(20, 20, f"focus = {focus[k]:.3f} [m]", fontsize=14, color='green')

        if roi is not None:
            x, y, w, h = roi
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='blue', fill=False))


        img   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img   = cv2.resize(img, (N,M))
        images[:, :, k] = img
        #images = np.stack((images,img[:,:,np.newaxis]), axis = 2)
        #print(f"\rReading [{k+1:02d}/{P:02d}]", end="")

        plt.show(block=False)
        plt.pause(0.02)

    images = np.array(images)
    return images, focus



def carve_depthmap(z, r, threshold=20):
    """
    Carves the depth map by removing pixels with reliability below the threshold.

    Args:
        z: The depth map.
        r: The reliability map.
        threshold: The reliability threshold (default: 20 dB).

    Returns:
        The carved depth map.
    """

    zc = z.copy()
    zc[r < threshold] = np.nan
    return zc

def display_depthmap(z, zc):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(z, cmap='copper')
    plt.title('Full Depthmap')

    plt.subplot(1, 2, 2)
    plt.imshow(zc, cmap='copper')
    plt.title('Carved Depthmap (R < 20 dB)')

    plt.show()

# -------------------------
def main():
    "main algo"

    opts = {
    #     'RGB': len(im.shape) == 3,
         'interp': True,
         'fmeasure': 'UDUD', #'LAPM',
         'filter': 0,
         'nhsize': 5,
         'focus': list(range(1, 30 + 1))
    #     'ROI': args.roi,
    #     'size': (im.shape[0], im.shape[1], len(imlist))
    }    

    fpath       = r'.\trial\sff\imdata.mat'
    imdata      = load_matdata(fpath)
    images, focus = read_show_images(imdata)

    opts['focus'] = focus

    #  [z, r] = sff(imdata.images, 'focus', imdata.focus)
    z, r        = compute_sff(images, opts)
    zc          = carve_depthmap(z, r, threshold=20)
    display_depthmap(z, zc)

# -------------------------
def main_razy():
    "main algo with dta from Razy"

    opts = {
    #     'RGB': len(im.shape) == 3,
         'interp': True,
         'fmeasure': 'UDUD', #'LAPM',
         'filter': 0,
         'nhsize': 5,
         'focus': list(range(1, 30 + 1))
    #     'ROI': args.roi,
    #     'size': (im.shape[0], im.shape[1], len(imlist))
    }    

    fpath       = r'C:\Data\Focus\Razy\Cups\*.JPG'
    images, focus = read_razy_images(fpath)

    opts['focus'] = focus

    #  [z, r] = sff(imdata.images, 'focus', imdata.focus)
    z, r        = compute_sff(images, opts)
    zc          = carve_depthmap(z, r, threshold=20)
    display_depthmap(z, zc)    
   
if __name__ == '__main__':
    #main()
    main_razy()