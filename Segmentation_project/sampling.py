import sys
import torch
import numpy as np
from random import randint
from torch.utils import data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from nets.zoomout import Zoomout
from data.loader import PascalVOC
from utils import *
import gc

def extract_samples(zoomout, dataset):
    """
    TODO: Follow the directions in the README
    to extract a dataset of 1x1xZ features along with their labels.
    Predict from zoomout using:
         with torch.no_grad():
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))
    """

# 1. For each training image, sample 3 pixels from each class
# 2. For each pixel, save the pixel feature (i.e. 1x1xZ) dimension along with the identity of that pixel as a label
# 3. There are an average of 2.5 classes per image so this process should yield around 10,000 example (feature,label) pairs
    features_labels =[]
    features = []
    for image_idx in range(len(dataset)):
        images, labels = dataset[image_idx]
        classes = np.unique(labels)
        dict1 = {}
        temp = {}
        ## create each class in dictionary
        for index in range(len(classes)):
            temp[classes[index]] = 0
            dict1[classes[index]] = []
        while (all(value == 3 for value in temp.values()) == False):
            i = randint(0, len(labels)-1)
            j = randint(0,len(labels[0])-1)
            ## if we need more sample, add it and record its index
            if temp[int(labels[i][j])]<3:
                temp[int(labels[i][j])] +=1
                dict1[int(labels[i][j])].append((i,j))
        ## if got all samples, break
        ## by now, we get 3 pixels for each class, and next we need to use zoomout.
        
        for key,item in dict1.items():
            for i in range(len(item)):
                features_labels.append(key)
                ## optional normalize the data
                # images = images.float()/255
                # images = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(images)
                with torch.no_grad():
                     feature = zoomout(images.cpu().float().unsqueeze(0))
                ## this should return 1472*224*224
                temp1 = feature[:,:,item[i][0],item[i][1]]
                features.append(temp1.detach().clone().numpy())
        print("image"+str(image_idx))
    return features, features_labels


def main():
    zoomout = Zoomout().cpu().float()
    for param in zoomout.parameters():
        param.requires_grad = False
    dataset_train = PascalVOC(split = 'train')
    features, labels = extract_samples(zoomout, dataset_train)
    np.save("./features/feats_x.npy", features)
    np.save("./features/feats_y.npy", labels)
    dataset_x = features
    means = np.mean(dataset_x,dim=0)
    stds = np.std(dataset_x,dim=0)
    np.save("./features/mean.npy", means)
    np.save("./features/std.npy", stds)


if __name__ == '__main__':
    main()
