import os
import collections
import json

import os.path as osp
import numpy as np
from PIL import Image
import PIL
import collections
import torch
import torchvision
import imageio
from torch.utils import data
# from scipy.misc import imsave
from torch.utils import data
import random 
from torch.autograd import Variable

pascal_labels = np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
        [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
        [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
        [0,192,0], [128,192,0], [0,64,128]])


class PascalVOC(data.Dataset):
    def __init__(self,root='./data/VOCdevkit/VOC2012', split="train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.n_classes = 21
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.im_height = 224
        self.im_width = 224
        self.mean_pixel = np.array([103.939, 116.779, 123.68])

        self.files = collections.defaultdict(list)
        for split in ["train", "val"]:
            file_list = tuple(open(root + '/ImageSets/Segmentation/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/JPEGImages/' + img_name + '.jpg'
        label_path = self.root + '/SegmentationClass/' + img_name + '.png'

        pil_img = Image.open(img_path).convert('RGB')
        pil_lbl = Image.open(label_path).convert('P')

        pil_img = pil_img.resize((self.im_height, self.im_width), PIL.Image.BILINEAR)
        pil_lbl = pil_lbl.resize((self.im_height, self.im_width), PIL.Image.NEAREST)

        img = np.array(pil_img)
        lbl = np.array(pil_lbl)
        lbl[lbl==255] = 0

        # pil_img = 0
        # pil_lbl = 0

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        obj_ids = np.unique(lbl)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = lbl== obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.img_transform is not None:
            img, target = self.img_transform(img), self.label_transform(target)

        return torch.from_numpy(img), target


def visualize(path, predicted_label):
    label = predicted_label

    label_viz = np.zeros((label.shape[0], label.shape[1], 3))

    for unique_class in np.unique(label):
        if unique_class != 0:
            indices = np.argwhere(label==unique_class)
            for idx in range(indices.shape[0]):
                label_viz[indices[idx, 0], indices[idx, 1], :] = pascal_labels[unique_class,:]

    # imsave(path, label_viz)
    imageio.imwrite(path, label_viz)