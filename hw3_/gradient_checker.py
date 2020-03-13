#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def is_same(im1, im2, eps=0.01):
    return np.sum(np.abs(im1 - im2) / np.prod(im1.shape)) <= eps

def print_res_match(im1, im2, eps=0.01):
    print('Result Match = ' + str(is_same(im1, im2, eps)))

def print_shape_match(im1, im2):
    print('Shape Match = ' + str(im1.shape == im2.shape))

# import the reference code (master solution)
# and the student's code
student = __import__('hw3_pynet' if len(sys.argv) == 1 else sys.argv[1])

print('Testing Conv2d:')
# initialize Conv2d
student_conv = student.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
ref_conv = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1, bias=True)
# align input data
conv_data_pt = Variable(torch.rand(1, 3, 32, 32), requires_grad = True)
conv_data_numpy = conv_data_pt.data.numpy()
# align weight and bias
ref_conv.weight.data = torch.from_numpy(student_conv.weight).clone().detach().requires_grad_(True)
ref_conv.bias.data = torch.from_numpy(student_conv.bias).clone().detach().requires_grad_(True)
# check output
ref_out = ref_conv(conv_data_pt)
student_out = student_conv.forward(conv_data_numpy)
print_shape_match(ref_out.data.numpy(), student_out)
print_res_match(ref_out.data.numpy(), student_out)
# check gradients
grad_out, grad_w, grad_b = student_conv.backward(np.ones_like(student_out))
ref_out.backward(torch.ones_like(ref_out))
print_res_match(conv_data_pt.grad.data.numpy(), grad_out)
print_res_match(ref_conv.weight.grad.data.numpy(), grad_w)
print_res_match(ref_conv.bias.grad.data.numpy(), grad_b)
print('\n')

print('Testing Linear:')
# initialize Linear
student_linear = student.Linear(30, 10)
ref_linear = nn.Linear(30, 10, bias=True)
# align input data
linear_data_pt = Variable(torch.rand(1, 30), requires_grad = True)
linear_data_numpy = linear_data_pt.data.numpy()
# align weight and bias
# NOTE: Use transpose is because the slight shape definition difference between PyNet and Pytorch.
ref_linear.weight.data = torch.from_numpy(student_linear.weight.transpose().astype(np.float32)).clone().detach().requires_grad_(True)
ref_linear.bias.data = torch.from_numpy(student_linear.bias.transpose().astype(np.float32)).clone().detach().requires_grad_(True)
# check output
ref_out = ref_linear(linear_data_pt)
student_out = student_linear.forward(linear_data_numpy)
print_shape_match(ref_out.data.numpy(), student_out)
print_res_match(ref_out.data.numpy(), student_out)
# check gradients
grad_out, grad_w, grad_b = student_linear.backward(np.ones_like(student_out))
ref_out.backward(torch.ones_like(ref_out))
print_res_match(linear_data_pt.grad.data.numpy(), grad_out)
print_res_match(ref_linear.weight.grad.data.numpy().transpose(), grad_w)
print_res_match(ref_linear.bias.grad.data.numpy().transpose(), grad_b)
print('\n')

print('Testing MaxPool2d:')
# initialize Linear
student_maxpool = student.MaxPool2d(kernel_size=2, stride=2)
ref_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
# align input data
maxpool_data_pt = Variable(torch.rand(1, 3, 32, 32), requires_grad = True)
maxpool_data_numpy = maxpool_data_pt.data.numpy()
# check output
ref_out = ref_maxpool(maxpool_data_pt)
student_out = student_maxpool.forward(maxpool_data_numpy)
print_shape_match(ref_out.data.numpy(), student_out)
print_res_match(ref_out.data.numpy(), student_out)
# check gradients
grad_out = student_maxpool.backward(np.ones_like(student_out))
ref_out.backward(torch.ones_like(ref_out))
print_res_match(maxpool_data_pt.grad.data.numpy(), grad_out)
print('\n')

print('Testing ReLU:')
# initialize Relu
student_relu = student.ReLU()
ref_relu = nn.ReLU()
# align input data
relu_data_pt = Variable(torch.rand(1, 3, 32, 32), requires_grad = True)
relu_data_numpy = relu_data_pt.data.numpy()
# check output
ref_out = ref_relu(relu_data_pt)
student_out = student_relu.forward(relu_data_numpy)
print_shape_match(ref_out.data.numpy(), student_out)
print_res_match(ref_out.data.numpy(), student_out)
# check gradients
grad_out = student_relu.backward(np.ones_like(student_out))
ref_out.backward(torch.ones_like(ref_out))
print_res_match(relu_data_pt.grad.data.numpy(), grad_out)
print('\n')

print('Testing BatchNorm1d:')
# initialize BN
student_bn = student.BatchNorm1d(100)
ref_bn = nn.BatchNorm1d(100)
# align input data
bn_data_pt = Variable(torch.rand(20, 100), requires_grad = True)
bn_data_numpy = bn_data_pt.data.numpy()
# align BN params
ref_bn.weight.data = torch.from_numpy(student_bn.gamma).clone().detach().requires_grad_(True)
ref_bn.bias.data = torch.from_numpy(student_bn.beta).clone().detach().requires_grad_(True)
ref_bn.running_mean.data = torch.from_numpy(student_bn.r_mean).clone().detach().requires_grad_(False)
ref_bn.running_var.data = torch.from_numpy(student_bn.r_var).clone().detach().requires_grad_(False)
ref_bn.momentum = student_bn.momentum
ref_bn.eps = student_bn.eps
# check output
ref_out = ref_bn(bn_data_pt)
student_out = student_bn.forward(bn_data_numpy, train=True)
print_shape_match(ref_out.data.numpy(), student_out)
print_res_match(ref_out.data.numpy(), student_out)
# check gradients
grad_out, grad_gamma, grad_beta = student_bn.backward(np.ones_like(student_out))
ref_out.backward(torch.ones_like(ref_out))
print_res_match(bn_data_pt.grad.data.numpy(), grad_out)
print_res_match(ref_bn.weight.grad.data.numpy(), grad_gamma)
print_res_match(ref_bn.bias.grad.data.numpy(), grad_beta)
print('\n')

print('Testing cross_entropy_loss_with_softmax:')
# initialize cross_entropy loss
# Note: PyNet's cross_entropy loss functions as Pytorch
# CrossEntropyLoss when reduction='none', So the output loss shape as well
# as the gradients shape are N*C shape.
student_loss = student.CrossEntropyLossWithSoftmax()
ref_loss = nn.CrossEntropyLoss(reduction='none')
# align input data
loss_data_pt = torch.randn(3, 5, requires_grad = True)
loss_data_numpy = loss_data_pt.data.numpy()
# align target data
loss_target_pt = torch.empty(3, dtype=torch.long).random_(5)
loss_target_numpy = loss_target_pt.data.numpy()
# check output
ref_out = ref_loss(loss_data_pt, loss_target_pt)
student_out = student_loss.forward(loss_data_numpy, loss_target_numpy)
print_shape_match(ref_out.data.numpy(), student_out)
print_res_match(ref_out.data.numpy(), student_out)
# check gradients
grad_out = student_loss.backward(np.ones_like(student_out))
ref_out.backward(torch.ones_like(ref_out))
print_res_match(loss_data_pt.grad.data.numpy(), grad_out)
