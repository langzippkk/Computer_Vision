import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
import util2
import numpy as np
from vis import *

def evaluate(model, data_loader, device):
    counter = 0
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = util2.MetricLogger(delimiter="  ")
    header = 'Test:'
    labels = []
    sample_metrics = []
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        ##print(image[0].shape)
        image = image[0].view(1,3,224,224)
        with torch.no_grad():
            outputs = model(image.float())
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
            outputs = outputs[0]
            targets = targets[0]
            labels += targets['labels'].tolist()
        ##################################non-max and visualization#############################################
            outputs = util2.non_max_suppression(outputs)
        if (counter%100 == 0):
            display_images(image)
            # random_color = random_colors(1, bright=True)
            box = outputs['boxes']
            # print(box)
            # image_box = draw_box(image, box[0], random_color)
            # display_images(image_box)
            draw_boxes(image, boxes=box, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None)
        counter +=1
        sample_metrics += util2.get_batch_statistics(outputs, targets, iou_threshold=0.5)

        ## outputs will get 1. boxes 2. labels 3. scores 4.masks 
#################################################################################
    print(sample_metrics)
    device = torch.device('cpu')
    
    ## sample_metrics = [t.to(device) for t in sample_metrics]
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = util2.ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class
