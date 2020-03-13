import sys
import torch
import numpy as np
import pandas as pd
from random import randint
from torch.utils import data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from nets.zoomout import Zoomout
from data.loader import PascalVOC
from utils import *
import gc

def extract_samples(dataset):

    count = pd.Series(np.zeros(21),index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    for image_idx in range(len(dataset)):
        images, labels = dataset[image_idx]
        labels = labels.view(224*224)
        list1 =labels.data.tolist()
        labels = pd.Series(list1)
        temp = labels.value_counts()
        count = count.add(temp,fill_value =0)
        
    return count


def main():
    dataset_train = PascalVOC(split = 'train')
    stats = extract_samples(dataset_train)
    stats = 1/(np.sqrt(np.array(stats)/np.sum(stats)))
    np.save("./features/stats.npy", stats)


if __name__ == '__main__':
    main()
