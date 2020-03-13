"""
   Here you will implement a relatively shallow neural net classifier on top of the hypercolumn (zoomout) features.
   You can look at a sample MNIST classifier here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from zoomout import *
import numpy as np
from torchvision import transforms

class FCClassifier(nn.Module):
    """
        Fully connected classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, n_classes=21):
        super(FCClassifier, self).__init__()
        """
        TODO: Implement a fully connected classifier.
        """
        # You will need to compute these and store as *.npy files
        self.mean = torch.Tensor(np.load("./features/mean.npy"))
        self.std = torch.Tensor(np.load("./features/std.npy"))
        self.fc = nn.Linear(1472,368)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(368,92)
        self.ReLU1 = nn.ReLU()
        self.fc2 = nn.Linear(92,21)

    def forward(self, x):
        # normalization
        x = (x - self.mean)/self.std
        out = self.fc(x)
        out = self.ReLU(out)
        out1 = self.fc1(out)
        out1 = self.ReLU1(out1)
        out2 = self.fc2(out1)

        return out2

class DenseClassifier(nn.Module):
    """
        Convolutional classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, fc_model, n_classes=21):
        super(DenseClassifier, self).__init__()
        """
        TODO: Convert a fully connected classifier to 1x1 convolutional.
        """
        mean = torch.Tensor(np.load("./features/mean.npy"))
        std = torch.Tensor(np.load("./features/std.npy"))
        # You'll need to add these trailing dimensions so that it broadcasts correctly.
        self.mean = torch.Tensor(np.expand_dims(np.expand_dims(mean, -1), -1))
        self.std = torch.Tensor(np.expand_dims(np.expand_dims(std, -1), -1))
        weight1 = (fc_model.fc.weight)
        weight2 = (fc_model.fc1.weight)
        weight3 = (fc_model.fc2.weight)

        self.conv1 = nn.Conv2d(1472,368,1)
        self.conv2 = nn.Conv2d(368,92,1)
        self.conv3 = nn.Conv2d(92,n_classes,1)
        self.ReLU = nn.ReLU()
        self.ReLU1 = nn.ReLU()
        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(weight1.view(368,1472,1,1))
            self.conv2.weight =  torch.nn.Parameter(weight2.view(92,368,1,1))
            self.conv3.weight =  torch.nn.Parameter(weight3.view(n_classes,92,1,1))
        # self.features = nn.Sequential(
        #     nn.Conv2d(3,192,5,padding = 2),
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(192, 192, 3, padding=1),
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(192,n_classes,1),
        #     nn.ReLU(inplace = True),
        #     ##nn.AvgPool2d(8,stride = 1)
        #     )

    def forward(self, x):
        """
        Make sure to upsample back to 224x224 --take a look at F.upsample_bilinear
        """
        # normalization
        x = (x - self.mean)/self.std
        out = self.conv1(x)
        out = self.ReLU(out)
        out1 = self.conv2(out)
        out1 = self.ReLU1(out1)
        out2 = self.conv3(out1)
        return out2

