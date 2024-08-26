"""
% Shape from focus (SFF)
% SINTAX:
%
%   Z  = sff(IMAGES)   
%   Z  = sff(IMAGES, option1, value1, option2, ...)
%   [Z,R] = sff(IMAGES, ...);
%
% DESCRIPTION:
% Estimate depthmap and reliability measure from defocused image 
% sequence using shape-from-focus (SFF). According to [2], depth for 
% pixels with R<20 dB is not reliable and should be descarded.
% 
% INPUTS:
% IMLIST,     is a 1xP cell a array where each cell is a 
%             string with the image path  corresponding to 
%             one frame of the focus sequence.
%
% Options and their values may be any of the following 
%(default value in parenthesis):
% 'fmeasure',   A string with the focus measure operator. (GLVM).
% 'filter',     A scalar with the size of median filter. (0).
% 'interp',     A logic flag to turn Gaussian interpolation on/off (true)
% 'nhsize',      An integer with the size of the focus mesure window. (9)
% 'focus',      A vector with the focus position of each image. (1:P)
% 'ROI',        Scene ROI as a rectangle [xo yo W H]. Default is [].
%
% OUTPUTS:
% Z,        is a MxN matrix with the reconstructed depthmap obtained using
%           SFF as originally proposed in [1]
% R,        is a MxN matrix with the reliability measure (in dB) of the
%           estimated depth computed as proposed in [2]
%
% References:
% [1] S.K. Nayar and Y. Nakagawa, PAMI, 16(8):824-831, 1994. 
% [2] S. Pertuz, D. Puig, M. A. Garcia, Reliability measure for shape-from
% focus, IMAVIS, 31:725734, 2013.
%
%S. Pertuz
%Jan/2016.

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


"""

import numpy as np
import time
import cv2
import argparse
from scipy.signal import convolve2d #, median_filter
from focus_measure import focus_measure
# -------------------------

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


# -------------------------
def compute_sff(images, opts):
    """
    Computes the shape from focus (SFF) depth map and reliability measure.

    Args:
        images: A numpy array of shape (M, N, P) containing the image sequence.
        opts: A dictionary containing options for the SFF algorithm.
        focus_measure_func: A function to compute the focus measure.

    Returns:
        A tuple (z, R) where z is the depth map and R is the reliability map.
    """

    M, N, P = images.shape

    # Compute focus measure volume
    start_time = time.time()
    fm = np.zeros((M, N, P))
    for p in range(P):
        
        #fm[:, :, p] = focus_measure_func(images[:, :, p], opts.fmeasure, opts.nhsize)
        fm[:, :, p] = focus_measure(images[:, :, p], opts['fmeasure'], opts['nhsize'])
        print(f"\rFmeasure [{(p+1):02d}/{P:02d}]", end="")
    print()
    end_time = time.time()
    print(f"Fmeasure time: {end_time - start_time:.2f} s")

    # Estimate depth map
    start_time = time.time()
    if opts['interp'] :  # Adjust for Python's nargout equivalent
        I, zi, s, A = gauss3P(opts['focus'], fm)
        z = zi
        z[z > np.max(opts['focus'])] = np.max(opts['focus'])
        z[z < np.min(opts['focus'])] = np.min(opts['focus'])
    else:
        I, zi, s, A = gauss3P(opts['focus'], fm)
        z = opts['focus'][I]
    # else:
    #     I = np.argmax(fm, axis=2)
    #     z = opts['focus'][I]

    fmax = opts['focus'][I]
    z[np.isnan(z)] = fmax[np.isnan(z)]
    end_time = time.time()
    print(f"Depthmap time: {end_time - start_time:.2f} s")

    # Median filter
    if opts['filter'] != 0:
        print("Smoothing")
        z = median_filter(z, size=(opts['filter'], opts['filter']))
        print("[100%]")

    # Reliability measure

    start_time = time.time()
    print("Rmeasure")
    err = np.zeros((M, N))

    # Compute fitting error
    for p in range(P):
        err += np.abs(fm[:, :, p] - A * np.exp(-((opts['focus'][p] - zi) ** 2) / (2 * s**2)))
        print(f"\rRmeasure [{(p+1):02d}/{P:02d}]", end="")
    print()

    h = np.ones((opts['nhsize'], opts['nhsize'])) / (opts['nhsize'] * opts['nhsize'])  # Equivalent to fspecial('average')
    err = convolve2d(err, h, mode='same')

    R = 20 * np.log10(P * fmax / err)
    mask = np.isnan(zi)
    R[mask | (R < 0) | np.isnan(R)] = 0
    print()
    end_time = time.time()
    print(f"Rmeasure time: {end_time - start_time:.2f} s")
    return z, R



# -------------------------

def parse_inputs(imlist, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=int, default=0, help='Size of median filter')
    parser.add_argument('--fmeasure', type=str, default='glvm', help='Focus measure operator')
    parser.add_argument('--focus', type=float, nargs='+', default=None, help='Focus positions')
    parser.add_argument('--interp', type=bool, default=True, help='Enable Gaussian interpolation')
    parser.add_argument('--nhsize', type=int, default=9, help='Size of focus measure window')
    parser.add_argument('--roi', type=int, nargs=4, default=None, help='Region of interest')

    args = parser.parse_args(args=[])  # Parse empty args to use kwargs

    # Convert kwargs to argparse namespace
    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # Determine image size
    import cv2
    im = cv2.imread(imlist[0])
    if im is None:
        raise ValueError("Failed to read image")
    
    opts = {
        'RGB': len(im.shape) == 3,
        'interp': args.interp,
        'fmeasure': args.fmeasure,
        'filter': args.filter,
        'nhsize': args.nhsize,
        'focus': args.focus if args.focus else list(range(1, len(imlist) + 1)),
        'ROI': args.roi,
        'size': (im.shape[0], im.shape[1], len(imlist))
    }

    return opts


# -------------------------
def gauss3P(x, Y):
    """
    Closed-form solution for Gaussian interpolation using 3 points

    Args:
        x: Independent variable data.
        Y: Dependent variable data.

    Returns:
        I: Index of the maximum value in each column of Y.
        u: Mean of the Gaussian distribution.
        s: Standard deviation of the Gaussian distribution.
        A: Amplitude of the Gaussian distribution.
    """

    STEP = 2
    M, N, P = Y.shape
    I = np.argmax(Y, axis=2)
    I = np.clip(I, STEP, P - STEP)  # Clip indices to avoid out-of-bounds

    IN, IM = np.meshgrid(np.arange(N), np.arange(M))
    Ic = I.flatten()
    Index1 = IM.flatten() * P * N + IN.flatten() * P + Ic - STEP
    Index2 = IM.flatten() * P * N + IN.flatten() * P + Ic
    Index3 = IM.flatten() * P * N + IN.flatten() * P + Ic + STEP

    x1 = x[Ic - STEP].reshape(M, N)
    x2 = x[Ic].reshape(M, N)
    x3 = x[Ic + STEP].reshape(M, N)
    y1 = np.log(Y.flat[Index1]).reshape(M, N)
    y2 = np.log(Y.flat[Index2]).reshape(M, N)
    y3 = np.log(Y.flat[Index3]).reshape(M, N)

    c = ((y1 - y2) * (x2 - x3) - (y2 - y3) * (x1 - x2)) / \
        ((x1**2 - x2**2) * (x2 - x3) - (x2**2 - x3**2) * (x1 - x2))
    b = ((y2 - y3) - c * (x2 - x3) * (x2 + x3)) / (x2 - x3)
    a = y1 - b * x1 - c * x1**2

    s = np.sqrt(-1 / (2 * c))
    u = b * s**2
    A = np.exp(a + u**2 / (2 * s**2))

    return I, u, s, A
