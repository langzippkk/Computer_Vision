"""
TODO: Implement zoomout feature extractor.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

class Zoomout(nn.Module):
    def __init__(self):
        super(Zoomout, self).__init__()

        # load the pre-trained ImageNet CNN and list out the layers
        self.vgg = models.vgg11(pretrained=True)
        self.feature_list = list(self.vgg.features.children())
        ##print(self.feature_list)
        self.RELU0 = nn.Sequential(*self.feature_list[:2])
        self.RELU1 = nn.Sequential(*self.feature_list[2:5])
        self.RELU2 = nn.Sequential(*self.feature_list[5:10])
        self.RELU3 = nn.Sequential(*self.feature_list[10:15])
        self.RELU4 = nn.Sequential(*self.feature_list[15:20])
        # self.features = nn.Sequential(self.RELU0,self.RELU1,self.RELU2,self.RELU3,self.RELU4,self.RELU5,self.RELU6)     
        ##print((self.feature_list))
        """
        TODO:  load the correct layers to extract zoomout features.
        """

    def forward(self, x):

        """
        TODO: load the correct layers to extract zoomout features.
        Hint: use F.upsample_bilinear and then torch.cat.
        """
        (N,C,W,H) = x.shape
        x = self.RELU0(x)
        result = F.upsample(x, size=(W,H), mode='bilinear')
        ## 64 channel
        x = self.RELU1(x)
        temp = F.upsample(x, size=(W,H), mode='bilinear')
        result = torch.cat((result,temp),dim = 1)

        x = self.RELU2(x)
        temp = F.upsample(x, size=(W,H), mode='bilinear')
        result = torch.cat((result,temp),dim = 1)


        x = self.RELU3(x)
        temp = F.upsample(x, size=(W,H), mode='bilinear')
        result = torch.cat((result,temp),dim = 1)

        x = self.RELU4(x)
        temp = F.upsample(x, size=(W,H), mode='bilinear')
        result = torch.cat((result,temp),dim = 1)
        # for i in range(1,len(self.features)):
        #     temp = F.upsample(x[i], size=(H,W,D), mode='bilinear')
        #     result = torch.cat((result,temp),dim = 2)
        return result