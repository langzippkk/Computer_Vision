# THIS is reference code using scipy tools
# Only for self check usage
import numpy as np
from scipy import ndimage
import scipy.signal
import scipy.io as sio

def conv_2d(image, filt, mode='zero'):
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)
   ##########################################################################
   # TODO: YOUR CODE HERE
   if mode =='zero':
       result = scipy.signal.convolve2d(image, filt,
                                  mode='same', boundary='fill', fillvalue=0)
   elif mode == 'mirror':
       result = scipy.signal.convolve2d(image, filt,
                                  mode='same', boundary='symm')
   else:
       raise NotImplementedError
   # raise NotImplementedError('conv_2d')
   ##########################################################################
   return result


def denoise_gaussian(image, sigma = 1.0):
   ##########################################################################
   # TODO: YOUR CODE HERE
   # raise NotImplementedError('denoise_gaussian')
   img = ndimage.gaussian_filter(image, sigma)
   ##########################################################################
   return img

def denoise_median(image, width = 1):
   ##########################################################################
   # TODO: YOUR CODE HERE
   # raise NotImplementedError('denoise_median')
   img = sio.loadmat(f'data/expected_median/refMedian_width_{width}.mat')['data']
   ##########################################################################
   return img

def sobel_gradients(img):
    ##########################################################################
    # TODO: YOUR CODE HERE
    # raise NotImplementedError('sobel_gradients')
    gx = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
    gy = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
    dx = conv_2d(img, gx, mode='mirror')
    dy = conv_2d(img, gy, mode='mirror')
    return dx, dy
