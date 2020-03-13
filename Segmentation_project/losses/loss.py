import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy2d(predicts, targets):
    """
    Hint: look at torch.nn.NLLLoss.  Considering weighting
    classes for better accuracy.
    """
    weight = torch.Tensor(np.load("./features/stats.npy"))
    ##weight = torch.Tensor(np.array([0.01,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.1,1,1,1,1,1]))
    m = torch.nn.LogSoftmax(dim=1)
    loss = torch.nn.NLLLoss(weight = weight)
    targets = targets.long()
    output = loss(m(predicts),targets)
    return output

def cross_entropy1d(predicts, targets):
    weight = torch.Tensor(np.load("./features/stats.npy"))
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    output = loss(m(predicts),targets)
    return output
    ##raise NotImplementedError