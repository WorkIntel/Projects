import numpy as np
from scipy.fftpack import fft2, ifft2

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

# Helper functions (assuming these are defined elsewhere)
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


if __name__ == "__main__":
  "testing block processing"

  a = np.arange(24*32).reshape(32,24) #np.ones([32, 24])
  b = np.zeros([8, 8])+2
  a1 = a.reshape(4, 8, 3, 8).transpose(0, 2, 1, 3)
  res = a1 + b
  res1 = res.transpose(0, 2, 1, 3).reshape(a.shape)

  print(a1.shape)
  print(res.shape)
  print(np.all(res1 == (a+2)))

  def plus(a,b):
    return a+b
  
  res2 = block_process(a,b, plus)
  print(np.all(res2 == (a+2)))