import numpy as np
import matplotlib.pyplot as plt
from stft2d import stft2d
from istft2d import istft2d
import cv2 as cv

def stft2d_test(window_size=32, corr_enabled=False, test_type=1, fig_num=2):
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
      image = np.random.randn(256, 256) * 30 + 128
  elif test_type == 2:  # Test simple image
      # Assuming 'circuit' is a predefined image
      image = cv.imread(r"C:\Data\Depth\RobotAngle\image_rgb_1004.png")
      image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  elif test_type == 3:  # Test different size
      image = np.random.randn(160, 192) * 10
      image[50:58, 50:58] = 180
  elif test_type == 11:  # Test patterns for correlation
      image_patch = np.ones((16, 16))
      image_patch[4:13, 4:13] = 0
      image = np.tile(image_patch, (8, 8))
      image = image * 128 + 10 + np.random.randn(*image.shape) * 8
  else:
      raise ValueError("Unknown TestType")

  # Perform STFT and ISTFT
  stft                  = stft2d(image, window_size, corr_enabled)
  reconstructed_image   = istft2d(stft, window_size, corr_enabled)

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


if __name__ == "__main__":
  stft2d_test(window_size=16, corr_enabled=False, test_type=3)