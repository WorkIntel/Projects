import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import triang
#from scipy.fftpack import ifft2
from block_processing import block_process, multiply, ifft_multiply, max_location

def istft2d(ImF, window_size=16, corr_enabled=False):
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

  # Construct window mask
  window_mask   = np.ones((window_size,window_size)) #np.outer(triang(window_size), triang(window_size))

  # Define translations
  translations = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * (window_size // 2)

  # For correlation
  if corr_enabled:
    translations = translations * 0  # no offsets


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
    plt.title('Corr %s peak at %s' %(str(i),str(peak_xy)))
    plt.colorbar() #orientation='horizontal')
    plt.show()

  # Handle real output if correlation adjustment was not used
  if not corr_enabled:
    image = np.real(image)

  return image