import pickle

import torch
import numpy as np

def read_pkl(path):
    return pickle.load(open(path, 'rb'))

def write_pkl(obj, path):
    return pickle.dump(obj, open(path, 'wb'))

def iou(pred, target):
    mask = pred == target
    unique_labs = torch.unique(target)
    total_iou = 0.0
    for lab in unique_labs:
        n_pred = torch.sum(pred == lab)
        n_gt = torch.sum(target == lab)
        n_intersect = torch.sum((target == lab) * mask)
        n_union = n_pred + n_gt - n_intersect
        if n_union == 0:
            total_iou += 1
        else:
            total_iou += n_intersect * 1.0 / n_union
    
    return total_iou / len(unique_labs)

