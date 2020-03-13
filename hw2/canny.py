import numpy as np

"""
   Mirror an image about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of the specified width created by mirroring the interior
"""
def mirror_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   # mirror top/bottom
   top    = image[:wx:,:]
   bottom = image[(sx-wx):,:]
   img = np.concatenate( \
      (top[::-1,:], image, bottom[::-1,:]), \
      axis=0 \
   )
   # mirror left/right
   left  = img[:,:wy]
   right = img[:,(sy-wy):]
   img = np.concatenate( \
      (left[:,::-1], img, right[:,::-1]), \
      axis=1 \
   )
   return img

"""
   Pad an image with zeros about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of zeros
"""
def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img

"""
   Remove the border of an image.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx - 2*wx, sy - 2*wy), extracted by
              removing a border of the specified width from the sides of the
              input image
"""
def trim_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
   return img

"""
   Return an approximation of a 1-dimensional Gaussian filter.

   The returned filter approximates:

   g(x) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2) / (2 * sigma^2) )

   for x in the range [-3*sigma, 3*sigma]
"""
def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

"""
   CONVOLUTION

   Convolve a 2D image with a 2D filter.

   Requirements:

   (1) Return a result the same size as the input image.

   (2) You may assume the filter has odd dimensions.

   (3) The result at location (x,y) in the output should correspond to
       aligning the center of the filter over location (x,y) in the input
       image.

   (4) When computing a product at locations where the filter extends beyond
       the defined image, treat missing terms as zero.  (Equivalently stated,
       treat the image as being padded with zeros around its border).

   You must write the code for the nested loops of the convolutions yourself,
   using only basic loop constructs, array indexing, multiplication, and
   addition operators.  You may not call any Python library routines that
   implement convolution.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
def conv_2d(image, filt):
   # make sure that both image and filter are 2D arrays
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)
   # get image and filter size
   sx, sy = image.shape
   sk, sl = filt.shape
   # pad image border by filter width
   wx = (sk - 1) // 2
   wy = (sl - 1) // 2
   image = pad_border(image, wx, wy)
   # intialize convolution result
   result = np.zeros(image.shape)
   # convolve
   for x in range(wx, sx + wx):
      for y in range(wy, sy + wy):
         for k in range(sk):
            for l in range(sl):
               result[x,y] = result[x,y] + \
                  image[x-wx+k, y-wy+l] * filt[sk-1-k, sl-1-l]
   # remove padding
   result = trim_border(result, wx, wy)
   return result

"""
   CONVOLUTION WITH GAUSSIAN

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   You may approximate the G(x,y) filter by computing it on a
   discrete grid for both x and y in the range [-3*sigma, 3*sigma].

   See the gaussian_1d function for reference.

   Note:
   (1) Remember that the Gaussian is a separable filter.
   (2) Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border.

   Arguments:
      image - a 2D numpy array
      sigma - standard deviation of the Gaussian

   Returns:
      img   - smoothed image, a 2D numpy array of the same shape as the input
"""
def conv_2d_gaussian(image, sigma = 1.0):
   # generate Gaussian filters
   fx = gaussian_1d(sigma)
   fy = np.transpose(fx)
   # pad image by mirroring
   width = (fx.shape[1] - 1) // 2
   img = mirror_border(image, width, width)
   # convolve
   img = conv_2d(conv_2d(img, fx), fy)
   # remove padding
   img = trim_border(img, width, width)
   return img

"""
   SOBEL GRADIENT OPERATOR

   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.

   The Sobel operator estimates gradients dx, dy, of an image I as:

         [  1  2  1 ]
   dx =  [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

         [ 1  0  -1 ]
   dy =  [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

   where (*) denotes convolution.

   Note:
      (1) Your implementation should be as efficient as possible.
      (2) Avoid creating artifacts along the border of the image.

   Arguments:
      image - a 2D numpy array

   Returns:
      dx    - gradient in x-direction at each point
              (a 2D numpy array, the same shape as the input image)
      dy    - gradient in y-direction at each point
              (a 2D numpy array, the same shape as the input image)

"""
def sobel_gradients(image):
   # make sure that image is a 2D array
   assert image.ndim == 2
   # define filters
   fx_a = np.transpose(np.atleast_2d([1,0,-1]))
   fx_b = np.atleast_2d([1,2,1])
   fy_a = np.transpose(np.atleast_2d([1,2,1]))
   fy_b = np.atleast_2d([1,0,-1])
   # pad image by mirroring
   img = mirror_border(image, 1, 1)
   # compute gradients via separable convolution
   dx = conv_2d(conv_2d(img, fx_a), fx_b)
   dy = conv_2d(conv_2d(img, fy_a), fy_b)
   # remove padding
   dx = trim_border(dx, 1, 1)
   dy = trim_border(dy, 1, 1)
   return dx, dy

"""
   NONMAXIMUM SUPPRESSION

   Given an estimate of edge strength (mag) and direction (theta) at each
   pixel, suppress edge responses that are not a local maximum along the
   direction perpendicular to the edge.

   Equivalently stated, the input edge magnitude (mag) represents an edge map
   that is thick (strong response in the vicinity of an edge).  We want a
   thinned edge map as output, in which edges are only 1 pixel wide.  This is
   accomplished by suppressing (setting to 0) the strength of any pixel that
   is not a local maximum.

   Note that the local maximum check for location (x,y) should be performed
   not in a patch surrounding (x,y), but along a line through (x,y)
   perpendicular to the direction of the edge at (x,y).

   A simple, and sufficient strategy is to check if:
      ((mag[x,y] > mag[x + ox, y + oy]) and (mag[x,y] >= mag[x - ox, y - oy]))
   or
      ((mag[x,y] >= mag[x + ox, y + oy]) and (mag[x,y] > mag[x - ox, y - oy]))
   where:
      (ox, oy) is an offset vector to the neighboring pixel in the direction
      perpendicular to edge direction at location (x, y)

   Arguments:
      mag    - a 2D numpy array, containing edge strength (magnitude)
      theta  - a 2D numpy array, containing edge direction in [0, 2*pi)

   Returns:
      nonmax - a 2D numpy array, containing edge strength (magnitude), where
               pixels that are not a local maximum of strength along an
               edge have been suppressed (assigned a strength of zero)
"""
def nonmax_suppress(mag, theta):
   # make sure that input is 2D
   assert mag.ndim == 2
   assert theta.ndim == 2
   # pad input
   mag   = pad_border(mag, 1, 1)
   theta = pad_border(theta, 1, 1)
   # compute unit vector in direction of gradient
   offset_x = np.round(np.cos(theta)).astype(int);
   offset_y = np.round(np.sin(theta)).astype(int);
   # nonmax suppress
   sx, sy = mag.shape
   nonmax = np.zeros((sx, sy))
   for x in range(1,sx-1):
      for y in range(1,sy-1):
         ox = offset_x[x,y]
         oy = offset_y[x,y]
         val   = mag[x,y]
         val_a = mag[x + ox, y + oy]
         val_b = mag[x - ox, y - oy]
         if (((val > val_a) and (val >= val_b)) or \
             ((val > val_b) and (val >= val_a))):
            nonmax[x,y] = mag[x,y]
   # remove padding
   nonmax = trim_border(nonmax, 1, 1)
   return nonmax

"""
   CANNY EDGE DETECTOR

   Canny edge detector with nonmax suppression.

   NOTE: Returned arguments changed since Homework 1.

   Given an input image:

   (1) Compute gradients in x- and y-directions at every location using the
       Sobel operator.  See sobel_gradients() above.

   (2) Estimate edge strength (gradient magnitude) and direction.

   (3) Perform nonmaximum suppression of the edge strength map, thinning it
       in the direction perpendicular to that of a local edge.
       See nonmax_suppress() above.

   Arguments:
      image    - a 2D numpy array

   Returns:
      mag      - edge strength after nonmaximum suppression
      theta    - edge orientation at each pixel
"""
def canny_nmax(image):
   # make sure that image is a 2D array
   assert image.ndim == 2
   # estimate gradients
   dx, dy = sobel_gradients(image)
   # compute gradient magnitude and direction
   mag   = np.sqrt((dx * dx) + (dy * dy))
   theta = np.arctan2(dy, dx)
   # nonmax suppress
   mag = nonmax_suppress(mag, theta)
   return mag, theta
