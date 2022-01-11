import numpy as np
import tarfile, sys, os, time, math
from scipy import misc
from sklearn import metrics
import copy as cp
from scipy.stats import entropy, wasserstein_distance
import os.path as path
import torch

BG = 1.0/(84*84)

def preprocess_batch(smap):
    N = smap.shape[0]
    smap = smap.view(N,-1) # flatten batch-wise
    smap = torch.div(smap, torch.sum(smap,1).view(N,1))
    return smap

def preprocess2_batch(smap):
    N = smap.shape[0]
    smap = smap.view(N,-1) # flatten batch-wise
    smap -= BG
    smap = torch.clamp(smap, 0, None)
    smap = torch.div(smap, torch.sum(smap,1).view(N,1))
    return smap


def computeKL_batch(saliency_map, gt_saliency_map):
    epsilon = 2.2204e-16 #MIT benchmark
    N = saliency_map.shape[0]
    
    saliency_map = torch.squeeze(saliency_map).float()
    gt_saliency_map = torch.squeeze(gt_saliency_map).float()
    assert(saliency_map.shape==gt_saliency_map.shape)
    
    saliency_map = preprocess_batch(saliency_map)
    saliency_map = torch.clamp(saliency_map, epsilon, None)
    saliency_map = torch.div(saliency_map, torch.sum(saliency_map,1).view(N,1))
    
    gt_saliency_map = preprocess2_batch(gt_saliency_map)
    gt_saliency_map = torch.clamp(gt_saliency_map, epsilon, None)
    gt_saliency_map = torch.div(gt_saliency_map, torch.sum(gt_saliency_map,1).view(N,1))
    
    kl = gt_saliency_map*torch.log(torch.div(gt_saliency_map,saliency_map))
    return torch.sum(kl,1)