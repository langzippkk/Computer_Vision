#!/usr/bin/env python
# -*- coding: utf-8 -*-
# self_checker: compare hw1.py (student version) and hw1_reference (using external libiary)
import sys
import numpy as np
import matplotlib.pyplot as plt

from util import *

def is_same(im1, im2, eps=0.1):
    return np.sum(np.abs(im1 - im2) / np.prod(im1.shape)) <= eps

def print_im_match(im1, im2, eps=0.1):
    print('Image Match = ' + str(is_same(im1, im2, eps)))

def print_shape_match(im1, im2):
    print('Shape Match = ' + str(im1.shape == im2.shape))

# import the reference code (master solution)
# and the student's code
ref = __import__('hw1_reference')
student = __import__('hw1')

# image to run tests on
image = load_image('data/69015.jpg')

# Test 1
print('Testing convolution:')

kernels = [
    np.ones([5, 5]),
]

for kernel in kernels:
    refIm  = ref.conv_2d(image, kernel.tolist())
    studentIm = student.conv_2d(image, kernel)
    studentIm = np.array(studentIm)
    status = 'Kernel size ' + str(kernel.shape) + '. Match = ' + str(is_same(refIm, studentIm))
    print(status)
print('\n')


# Test 2
image = load_image('data/69015.jpg')
print('Denoise with Gaussian:')
sigmas = [
    1.0,
]
for sigma in sigmas:
    refGauss = ref.denoise_gaussian(image, sigma)
    studentGauss = student.denoise_gaussian(image, sigma)
    studentGauss = np.array(studentGauss)
    print('Sigma = ' + str(sigma))
    print_shape_match(refGauss, studentGauss)
    print_im_match(refGauss, studentGauss)
print('\n')

# Test 3
image = load_image('data/143090_noisy.png')
print('Denoise with Median:')
widths = [1]
for width in widths:
    refMedian = ref.denoise_median(image, width)
    studentMedian = student.denoise_median(image, width)
    studentMedian = np.array(studentMedian)
    print('width = ' + str(width))
    print_shape_match(refMedian, studentMedian)
    print_im_match(refMedian, studentMedian, 0.02)
print('\n')

# Test 4
print('Checking Sobel gradients:')
refDx, refDy = ref.sobel_gradients(image)
studentDx, studentDy = student.sobel_gradients(image)

studentDx = np.array(studentDx)
studentDy = np.array(studentDy)

print('Checking dx:')
print_shape_match(refDx, studentDx)
print_im_match(refDx, studentDx)

print('Checking dy:')
print_shape_match(refDy, studentDy)
print_im_match(refDy, studentDy)

print('\n')
