'''
Created on Nov 22, 2020

@author: github
'''

import torch
import torch.cuda as cuda
import torch.nn as nn
import numpy as np


def gdl_loss(gen_frames, gt_frames, alpha=2, cuda=False):
    '''
    From original version on https://github.com/wileyw/VideoGAN/blob/master/loss_funs.py
    which was referenced from Deep multi-scale video prediction beyond mean square error paper    
    :param gen_frames: generated output tensors
    :param gt_frames: ground truth tensors
    :param alpha: The power to which each gradient term is raised.
    '''
    filter_x_values = np.array(
        [
            [[[-1, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[0, 0, 0]], [[-1, 1, 0]]],
        ],
        dtype=np.float32,
    )
    filter_x = nn.Conv2d(3, 3, (1, 3), padding=(0, 1))
    
    filter_y_values = np.array(
        [
            [[[-1], [1], [0]], [[0], [0], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[-1], [1], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[0], [0], [0]], [[-1], [1], [0]]],
        ],
        dtype=np.float32,
    )
    filter_y = nn.Conv2d(3, 3, (3, 1), padding=(1, 0))
        
    filter_x.weight = nn.Parameter(torch.from_numpy(filter_x_values))  # @UndefinedVariable
    filter_y.weight = nn.Parameter(torch.from_numpy(filter_y_values))  # @UndefinedVariable
    
    dtype = torch.FloatTensor if not cuda else torch.cuda.FloatTensor  # @UndefinedVariable
    filter_x = filter_x.type(dtype)
    filter_y = filter_y.type(dtype)
    
    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)
    
    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)  # @UndefinedVariable
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)  # @UndefinedVariable
    
    grad_total = torch.stack([grad_diff_x, grad_diff_y])  # @UndefinedVariable
    
    return torch.mean(grad_total)  # @UndefinedVariable


class GDL(nn.Module):
    '''
    Gradient different loss function
    Target: reduce motion blur 
    '''
    
    def __init__(self, cudaUsed=False):
        super(GDL, self).__init__()
        self.cudaUsed = cuda.is_available() and cudaUsed

    def forward(self, gen_frames, gt_frames):
        return gdl_loss(gen_frames, gt_frames, cuda=self.cudaUsed)
