import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from stft2d import stft2d
from istft2d import istft2d

def stft2d_test_corr(window_size=16, corr_enabled=True, test_type=6, fig_num=1):
  """
  Tests 2D STFT decomposition based on image correlation.

  Args:
      window_size: Size of the triangular window (default: 16).
      corr_enabled: Boolean flag enabling correlation adjustment (default: True).
      test_type: Integer specifying the test case (default: 6).
      fig_num: Figure number for visualization (default: 1).
  """

  # Select test case
  if test_type == 1:  # Test random image
      image1 = np.random.randn(128, 128) * 60 + 60
      shift = np.array([2, 2]) * 1
      image2 = np.roll(image1, shift, axis=(0, 1))
  elif test_type == 2:  # Test simple image
      # Assuming 'trees' is a predefined image
      image = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_1004.png")
      image1 = cv.cvtColor(image, cv.COLOR_RGB2GRAY)      
      shift = np.array([2, 2]) * 2
      image2 = np.roll(image1, shift, axis=(0, 1))
  elif test_type == 3:  # Small size
      image1 = np.zeros((64, 64))
      image1[4 * window_size // 2 : 4 * window_size // 2 + window_size,
             4 * window_size // 2 : 4 * window_size // 2 + window_size] = 100
      shift = np.array([2, 2]) * 1
      image2 = np.roll(image1, shift, axis=(0, 1))
  elif test_type == 4:  # Small size
      image1 = np.random.rand(40, 40)
      shift = np.array([2, 2]) * 1
      image2 = np.roll(image1, shift, axis=(0, 1))
  elif test_type == 5:  # Test different size
      image1 = np.zeros((123, 171)) * 3
      image1[50:58, 50:58] = 180
      shift = np.array([2, 2]) * 1
      image2 = np.roll(image1, shift, axis=(0, 1))
  elif test_type == 6:  # Test image
      image1 = np.array(rgb2gray(imread('cameraman.tif')))
      shift = np.array([2, 2]) * 2
      image2 = np.roll(image1, shift, axis=(0, 1))
  elif test_type == 11:  # Test patterns for correlation
      image_patch = np.ones((16, 16))
      image_patch[4:13, 4:13] = 0
      image = np.tile(image_patch, (8, 8))
      image1 = image * 128 + 10 + np.random.randn(*image.shape) * 8
      shift = np.array([2, 2]) * 1
      image2 = np.roll(image1, shift, axis=(0, 1))
  elif test_type == 12:  # Test pattern against image
      image1    = np.random.randn(128, 128) * 60 + 50
      offset    = 36
      image_patch = image1[offset:offset+window_size,offset:offset+window_size]
      image2    = np.tile(image_patch, (8, 8))
      shift     = np.array([2, 2]) * 0
      image1    = np.roll(image1, shift, axis=(0, 1))      
  else:
      raise ValueError("Unknown TestType")

  # Check image sizes
  if image1.shape != image2.shape:
    raise ValueError("Images must be of the same size")

  # Perform STFT
  image1_stft = stft2d(image1, window_size, corr_enabled)
  image2_stft = stft2d(image2, window_size, corr_enabled)

  # Correlate and perform ISTFT
  correlation_stft = image1_stft * np.conj(image2_stft)
  correlation_image = istft2d(correlation_stft, window_size, corr_enabled)

  # Visualize correlation
  plt.figure(fig_num)
  #plt.imshow(np.log10(np.abs(correlation_image)), cmap='gray')
  plt.imshow(np.abs(correlation_image), cmap='gray')
  plt.title(f"Correlation Shift: {shift}")
  plt.colorbar(orientation='horizontal')
  plt.show()

if __name__ == "__main__":
  stft2d_test_corr(window_size=16, corr_enabled=True, test_type=12)