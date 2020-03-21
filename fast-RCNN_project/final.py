import os
import numpy as np
from eval import *
import torch.utils.data
from PIL import Image
##from evaluate import evaluate
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import torchvision
from data.loader import *
import util2
from train_one_epoch import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.datasets as dset
import torchvision.transforms as T
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if __name__ == '__main__':
  num_classes = 2  # 1 class (person) + background
  backbone = torchvision.models.mobilenet_v2(pretrained=True).features
  backbone.out_channels = 1280
   
  anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                     aspect_ratios=((0.5, 1.0, 2.0),))
   
  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                  output_size=7,
                                                  sampling_ratio=2)
  # put the pieces together inside a FasterRCNN model
  model = FasterRCNN(backbone,
                     num_classes=2,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler)

  model.parameters
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
      # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  # construct an optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.0005,momentum=0.9, weight_decay=0.0005)

  # and a learning rate scheduler which decreases the learning rate by
  # 10x every 3 epochs
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=3,
                                                 gamma=0.1)
#########################################################################3
  dataset_train = PascalVOC(split = 'train')
  dataset_val = PascalVOC(split = 'val')
  NUM_TRAIN = 4900
  train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4,collate_fn=util2.collate_fn)
  data_loader_test = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4,collate_fn=util2.collate_fn)
  num_classes = 2
  # get the model using our helper function
  # move model to the right device
  model.to(device)
  # let's train it for 10 epochs
  num_epochs = 1
  for epoch in range(num_epochs):
      # train for one epoch, printing every 10 iterations
      ##train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
      # update the learning rate
      lr_scheduler.step()
      # evaluate on the test dataset
      ##torch.save(model, "./models/RCNN.pkl")
      model = torch.load("./models/RCNN.pkl")
      precision, recall, AP, f1, ap_class = evaluate(model, data_loader_test, device=device)
      print("####################final###################################################")
      print(precision, recall, AP, f1, ap_class)

