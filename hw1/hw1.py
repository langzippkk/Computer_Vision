import numpy as np
import math
from PIL import Image

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
   CONVOLUTION IMPLEMENTATION (10 Points)

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
      mode   - 'zero': preprocess using pad_border or 'mirror': preprocess using mirror_border.

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
def conv_2d(image, filt, mode='zero'):
   # make sure that both image and filter are 2D arrays
    assert image.ndim == 2, 'image should be grayscale'
    filt = np.atleast_2d(filt)
    filt = np.flip(filt)
       ##########################################################################
       # TODO: YOUR CODE HERE
    image_h =int(image.shape[0])
    image_w = int(image.shape[1])
    kernel_h = int(filt.shape[0])
    kernel_w =int(filt.shape[1])
    if mode == 'zero':
        image = pad_border(image, wx = int((kernel_h-1)/2), wy = int((kernel_w-1)/2))
    if mode == 'mirror':
        image = mirror_border(image, wx = int((kernel_h-1)/2), wy = int((kernel_w-1)/2))
    result = np.zeros((image_h,image_w))
    
    for i in range(image_h):
        for j in range(image_w):
            result[i][j] = np.sum(filt*image[i:i+kernel_h,j:j+kernel_w])
 
    ## raise NotImplementedError('conv_2d')
       ##########################################################################
    return result

"""
   GAUSSIAN DENOISING (5 Points)

   Denoise an image by convolving it with a 2D Gaussian filter.

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
      img   - denoised image, a 2D numpy array of the same shape as the input
"""
def denoise_gaussian(image, sigma = 1.0):
   ##########################################################################
   # TODO: YOUR CODE HERE
    sigma = int(sigma)
    x = gaussian_1d(sigma = sigma )
    gaus_2d = x*(x.transpose())
    img = conv_2d(image,gaus_2d,mode = 'mirror')
    
   ##########################################################################
    return img

"""
   MEDIAN DENOISING (5 Points)

   Denoise an image by applying a median filter.

   Note:
       Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border. No padding needed in median denosing.

   Arguments:
      image - a 2D numpy array
      width - width of the median filter; compute the median over 2D patches of
              size (2*width +1) by (2*width + 1)

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input

 first trial that does not considering the boundary condition:
def denoise_median(image, width = 1):
   ##########################################################################
    m = 2*width+1
    
    
    image_h =int(image.shape[0])
    image_w = int(image.shape[1])
    result = np.zeros((image_h,image_w))
    
    for j in range(image_w-m+1):
         for i in range(image_h-m+1):
            temp = []
            for x in range(m):
                for y in range(m):
                    temp.append(image[i+x][j+y])
            median = (np.sort(temp))[int(m*m/2)]
            for x in range(m):
                for y in range(m):
                    result[i+x][j+y] = median
    img = result

   ##  raise NotImplementedError('denoise_median')
   ##########################################################################
    return img
"""
def denoise_median(image, width = 1):
   ##########################################################################
   # TODO: YOUR CODE HERE
   # discuss the boundary condition with Cheng Du
   
    [image_w, image_h] = np.shape(image)
    m = 2*width+1
    img = np.zeros((image_w, image_h))
    for i in range(image_w):
        for j in range(image_h):
            if  i+width>image_w:
                right = min(i+width, image_w)
                left = right-m
            elif i-width<0:
                left = max(i-width, 0)
                right = left+m
            else:
                left = i-width
                right = left+m
            if  j+width>image_h:
                bottom = min(j+width, image_h)
                top = bottom-m
            elif j-width <0:
                top = max(j-width, 0)
                bottom = top+m
            else:
                top = j-width
                bottom = top+m
        
            window = np.sort(image[left:right, top:bottom].flatten())
            img[i, j]= window[int(np.floor(len(window)/2))]
        
        
    return img
"""
   SOBEL GRADIENT OPERATOR (5 Points)
   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.
   The Sobel operator estimates gradients dx(horizontal), dy(vertical), of
   an image I as:

         [ 1  0  -1 ]
   dx =  [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

         [  1  2  1 ]
   dy =  [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

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
   ##########################################################################
   # TODO: YOUR CODE HERE
    dx_filter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    dy_filter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    dx = conv_2d(image, dx_filter, mode='mirror')
    dy = conv_2d(image,dy_filter,mode='mirror')
   ##########################################################################

    return dx, dy




"""
   NONMAXIMUM SUPPRESSION (10 Points)

   Nonmaximum suppression.

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
def direction(image):
    x,y = sobel_gradients(image)
    mag = np.sqrt(x**2+y**2)
    direction = (180*np.arctan(y/x))/math.pi
    direction += 180
    return direction,mag

def nonmax_suppress(mag, theta):
   ##########################################################################
   # TODO: YOUR CODE HERE
    m = len(mag)
    n = len(mag[1])
    mask = mag
    for i in range(1,m-1):
        for j in range(1,n-1):
            angle = theta[i][j]
    ## vertical
            if (angle>67.5 and angle<=112.5) or (angle>=247.5 and angle<=292.5): 
                if (mag[i-1][j])> mag[i][j] or (mag[i+1][j]> mag[i][j]):
                    mask[i][j]=0

    ## horizontal
            elif (angle>340 and angle<=22.5) or (angle>157.5 and angle<=202.5): 
                if ((mag[i][j+1])>mag[i][j] or (mag[i][j-1]> mag[i][j])):
                    mask[i][j]=0

    ## 45
            elif (angle>22.5 and angle<=67.5) or (angle > 202.5 and angle<247.5): 
                if (mag[i-1][j-1]>mag[i][j] or (mag[i+1][j+1]> mag[i][j])):
                    mask[i][j]=0

    ## 135
            else:
                if ((mag[i-1][j+1])>mag[i][j] or (mag[i+1][j-1]> mag[i][j])):
                    
                    mask[i][j]=0

    nonmax = mask
   ##########################################################################
    return nonmax

"""
   HYSTERESIS EDGE LINKING (10 Points)

   Hysteresis edge linking.

   Given an edge magnitude map (mag) which is thinned by nonmaximum suppression,
   first compute the low threshold and high threshold so that any pixel below
   low threshold will be thrown away, and any pixel above high threshold is
   a strong edge and will be preserved in the final edge map.  The pixels that
   fall in-between are considered as weak edges.  We then add weak edges to
   true edges if they connect to a strong edge along the gradient direction.

   Since the thresholds are highly dependent on the statistics of the edge
   magnitude distribution, we recommend to consider features like maximum edge
   magnitude or the edge magnitude histogram in order to compute the high
   threshold.  Heuristically, once the high threshod is fixed, you may set the
   low threshold to be propotional to the high threshold.

   Note that the thresholds critically determine the quality of the final edges.
   You need to carefully tuned your threshold strategy to get decent
   performance on real images.

   For the edge linking, the weak edges caused by true edges will connect up
   with a neighbouring strong edge pixel.  To track theses edges, we
   investigate the 8 neighbours of strong edges.  Once we find the weak edges,
   located along strong edges' gradient direction, we will mark them as strong
   edges.  You can adopt the same gradient checking strategy used in nonmaximum
   suppression.  This process repeats util we check all strong edges.

   In practice, we use a queue to implement edge linking.  In python, we could
   use a list and its fuction .append or .pop to enqueue or dequeue.

   Arguments:
     nonmax - a 2D numpy array, containing edge strength (magnitude) which is thined by nonmaximum suppression
     theta  - a 2D numpy array, containing edeg direction in [0, 2*pi)

   Returns:
     edge   - a 2D numpy array, containing edges map where the edge pixel is 1 and 0 otherwise.
"""
## recursion
def threshold(image, lowRatio=0.05, highRatio=0.2,weak = 25,strong= 255):
    high = image.max() * highRatio;
    low = high*lowRatio;
    m, n = image.shape
    threshold = np.zeros((m,n),dtype = float)
    s_i, s_j = np.where(image >= high)
    w_i, w_j = np.where((image <= high) & (image >= low))
    threshold[s_i, s_j] = strong
    threshold[w_i, w_j] = weak
    return threshold,weak,strong

def findweakedges(angle,threshold,direction,i,j,strong,weak):
    threshold[i][j] = strong
    row,col = threshold.shape
    if (i == 0 or i ==(row-1) or j==0 or j==(col-1)):
        return threshold
    if (angle>67.5 and angle<=112.5) and threshold[i-1][j] ==weak:
        findweakedges(angle,threshold,direction,i-1,j,strong,weak)
    if (angle>=247.5 and angle<=292.5) and threshold[i+1][j] ==weak and (i+1)<row:
        findweakedges(angle,threshold,direction,i+1,j,strong,weak)
    if (angle>= 340 and angle<=22.5) and threshold[i][j-1] ==weak:
        findweakedges(angle,threshold,direction,i,j-1,strong,weak)
    if (angle>= 157.5 and angle<=202.5) and threshold[i][j+1] ==weak and (j+1)<col:
        findweakedges(angle,threshold,direction,i,j+1,strong,weak)
    if (angle> 22.5 and angle<=67.5) and threshold[i-1][j+1] ==weak and (j+1)<col:
        findweakedges(angle,threshold,direction,i-1,j+1,strong,weak)
    if (angle> 202.5 and angle<=247.5) and threshold[i+1][j-1] ==weak and (i+1)<row:
        findweakedges(angle,threshold,direction,i+1,j-1,strong,weak)
    if (angle>= 112.5 and angle<=157.5) and threshold[i-1][j-1] ==weak:
        findweakedges(angle,threshold,direction,i-1,j-1,strong,weak)
    if (angle> 22.5 and angle<=67.5) and threshold[i+1][j+1] ==weak and (j+1)<col and (i+1)<row:
        findweakedges(angle,threshold,direction,i+1,j+1,strong,weak)
    else:
        return threshold

def hysteresis_edge_linking(nonmax, theta):
    ##########################################################################
    # TODO: YOUR CODE HERE
    row,col = nonmax.shape
    threshold1,weak,strong = threshold(nonmax)
    test = threshold1.copy()
    for i in range(1,row-1):
        for j in range(1,col-1):
            angle = theta[i][j]
            if threshold1[i][j] == strong:
                edge1 = findweakedges(angle,threshold1,direction,i,j,strong,weak)
    edge1[edge1>=strong] = strong
    edge1[edge1 < strong] = 0
    return edge1

"""
   CANNY EDGE DETECTOR (5 Points)

   Canny edge detector.

   Given an input image:

   (1) Compute gradients in x- and y-directions at every location using the
       Sobel operator.  See sobel_gradients() above.

   (2) Estimate edge strength (gradient magnitude) and direction.

   (3) Perform nonmaximum suppression of the edge strength map, thinning it
       in the direction perpendicular to that of a local edge.
       See nonmax_suppress() above.

   (4) Compute the high threshold and low threshold of edge strength map
       to classify the pixels as strong edges, weak edges and non edges.
       Then link weak edges to strong edges

   Return the original edge strength estimate (max), the edge
   strength map after nonmaximum suppression (nonmax) and the edge map
   after edge linking (edge)

   Arguments:
      image    - a 2D numpy array

   Returns:
      mag      - a 2D numpy array, same shape as input, edge strength at each pixel
      nonmax   - a 2D numpy array, same shape as input, edge strength after nonmaximum suppression
      edge     - a 2D numpy array, same shape as input, edges map where edge pixel is 1 and 0 otherwise.
"""
def canny(image):
   ##########################################################################
   # TODO: YOUR CODE HERE
   ## raise NotImplementedError('canny')
   ##########################################################################
# see the direction function:(1) Compute gradients in x- and y-directions at every location using the
#  Sobel operator.  See sobel_gradients() above.
# (2) Estimate edge strength (gradient magnitude) and direction.  
    theta,mag = direction(image)
    
# (3) Perform nonmaximum suppression of the edge strength map, thinning it
# in the direction perpendicular to that of a local edge.
    nonmax = nonmax_suppress(mag,theta)
    
# (4) Compute the high threshold and low threshold of edge strength map
#  to classify the pixels as strong edges, weak edges and non edges.
#  Then link weak edges to strong edges
    edge = hysteresis_edge_linking(nonmax, theta)
    return mag, nonmax, edge


# Extra Credits:
# (a) Improve Edge detection image quality (5 Points)
# (b) Bilateral filtering (5 Points)
# You can do either one and the maximum extra credits you can get is 5.
"""
    BILATERAL DENOISING (Extra Credits: 5 Points)
    Denoise an image by applying a bilateral filter
    Note:
        Performs standard bilateral filtering of an input image.
        Reference link: https://en.wikipedia.org/wiki/Bilateral_filter

        Basically, the idea is adding an additional edge term to Guassian filter
        described above.

        The weighted average pixels:

        BF[I]_p = 1/(W_p)sum_{q in S}G_s(||p-q||)G_r(|I_p-I_q|)I_q

        In which, 1/(W_p) is normalize factor, G_s(||p-q||) is spatial Guassian
        term, G_r(|I_p-I_q|) is range Guassian term.

        We only require you to implement the grayscale version, which means I_p
        and I_q is image intensity.

    Arguments:
        image       - input image
        sigma_s     - spatial param (pixels), spatial extent of the kernel,
                       size of the considered neighborhood.
        sigma_r     - range param (no normalized, a propotion of 0-255),
                       denotes minimum amplitude of an edge
    Returns:
        img   - denoised image, a 2D numpy array of the same shape as the input
"""

def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))

def bilateral_filter(res,image,filter_width,x,y,sigma_s,sigma_r):
    temp= Wp= 0
    m = 2*filter_width + 1
    for i in range(filter_width,m):
        for j in range(filter_width,m):
            x_left = x+i-filter_width
            y_left = y+j-filter_width          
            if x_left >= len(image):
                x_left = len(image)-1
            if y_left>= len(image[0]):
                y_left = len(image[0])-1
            gi = gaussian(image[x_left][y_left] - image[x][y], sigma_r)
            gs = gaussian((np.sqrt((x_left-x)**2+(y_left-y)**2)), sigma_s)
            temp += image[x_left][y_left] * gi * gs
            Wp += gi * gs
    res[x][y] = temp / Wp


def denoise_bilateral(image, sigma_s=1, sigma_r=25.5):
    assert image.ndim == 2, 'image should be grayscale'
    ##########################################################################
    # TODO: YOUR CODE HERE
    img = np.zeros(image.shape)
    filter_width = int(np.ceil(3.0 * sigma_s))
    for i in range(len(image)):
        for j in range(len(image[0])):
            bilateral_filter(img,image,filter_width,i,j,sigma_s,sigma_r)
    ##########################################################################
    return img


#####################################################################################