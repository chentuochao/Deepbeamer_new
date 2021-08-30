import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
import numpy as np
import asteroid #mimihao: 936838085 

#------------loss function --------------

class SISDRLoss(nn.Module):
    def __init__(self, offset, l=1e-3):
        super().__init__()
        
        self.l=l
        self.offset=offset
        
    def forward(self, signal, gt):
        return torch.sum(asteroid.losses.pairwise_neg_sisdr(signal[..., self.offset:], gt[..., self.offset:]))+self.l*torch.sum(signal**2)/signal.shape[-1]          
    
class L1Loss(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset=offset
        self.loss=nn.L1Loss()
        
    def forward(self, signal, gt):
        return self.loss(signal[..., self.offset:], gt[..., self.offset:])
    
    
class FuseLoss(nn.Module):
    def __init__(self, offset, r=50):
        super().__init__()
        self.offset=offset
        self.l1loss=nn.L1Loss()
        self.sisdrloss=asteroid.losses.pairwise_neg_sisdr
        self.r=r
        
    def forward(self, signal, gt):
        #print(signal.shape, gt.shape)
        if len(signal.shape)==2:
            signal=signal.unsqueeze(1)
            gt = gt.unsqueeze(1)
        a = signal[..., self.offset:]

        b = gt[..., self.offset:]*self.r + torch.mean(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
        #print(a)
        #print(b)
        #print(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
        #print(self.l1loss(a, b))
        #print("----------------------------")
        return self.l1loss(signal[..., self.offset:], gt[..., self.offset:])*self.r+torch.mean(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
