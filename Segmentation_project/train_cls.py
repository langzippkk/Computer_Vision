import sys
import torch
import argparse
import numpy as np
from PIL import Image
import json
import random
##from scipy.misc import toimage, imsave

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
import torchvision.transforms as transforms

from losses.loss import *
from nets.classifier import FCClassifier

from data.loader import PascalVOC
import torch.optim as optim
from utils import *

USE_GPU = False
dtype = torch.float32 # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

##epochs = 5

def train(dataset, model, optimizer, epoch):
    """
    TODO: Implement training for simple FC classifier.
        Input: Z-dimensional vector
        Output: label.
    """
    batch_size = 1000
    data_x, data_y = dataset
    model.train()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    t = 0
    for x, y in zip(data_x,data_y):
        x = torch.Tensor(x)
        y = torch.Tensor([y])
        outputs = model(x)
        loss = cross_entropy2d(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t +=1
        if (t%200 == 0):
            print(loss)
#######################################################
    torch.save(model, "./models/fc_cls.pkl")



def main():

    classifier = FCClassifier().float()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)# pick an optimizer.
    dataset_x = np.load("./features/feats_x.npy")
    dataset_y = np.load("./features/feats_y.npy")
    num_epochs = 10
    for epoch in range(num_epochs):
        print(epoch)
        train([dataset_x, dataset_y], classifier, optimizer, epoch)

if __name__ == '__main__':
    main()
