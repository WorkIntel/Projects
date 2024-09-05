import numpy as np
from scipy.fftpack import fft2, fftshift
from scipy.signal.windows import triang
import cv2 as cv
from block_processing import block_process, multiply, fft_multiply




def stft2d(Im, window_size=16, corr_enabled=False):
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
  Im            = Im.astype(np.float64)


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

  # Construct window mask
  window_mask = np.outer(triang(window_size), triang(window_size))
  #mask        = np.tile(window_mask, (row_win_num, col_win_num))

  # Construct frequency mask for correlation adjustment (if enabled)
  dc_mask     = np.ones((window_size, window_size))
  if corr_enabled :
    #center_idx = window_size // 2
    #dc_mask[center_idx-1:center_idx+1, center_idx-1:center_idx+1] = 0
    dc_mask[0,0]             = 0
    #dc_mask[window_size-1,0] = 0
    #dc_mask[0,window_size-1] = 0
    #dc_mask[window_size-1,window_size-1] = 0
    #pass

  frequency_mask      = dc_mask #fftshift(dc_mask) # np.tile(fftshift(dc_mask), (row_win_num, col_win_num))

  # Initialize STFT output
  stft              = np.zeros((n_rows, n_cols, 4), dtype=np.complex64)

  # Define translations
  translations      = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * (window_size // 2)

  # Loop through translations and perform STFT
  for i, translation in enumerate(translations):
    #row_indices   = np.arange(active_rows) + translation[0]
    #col_indices   = np.arange(active_cols) + translation[1]

    image_patch   = Im[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols] #

    image_patch   = block_process(image_patch, window_mask, multiply)
    fft_patch     = block_process(image_patch, frequency_mask, fft_multiply)

    #stft[translation[0]:translation[0]+active_rows, translation[1]:translation[1]+active_cols, i] = fft_patch
    stft[0:active_rows, 0:active_cols, i] = fft_patch

  return stft

if __name__ == "__main__":
  "testing "
  image   = np.random.randn(256, 256) * 30 + 128
  imgfft  = stft2d(image)
  print(imgfft.shape)