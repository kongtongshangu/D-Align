import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from torchvision.transforms.functional import center_crop
logger = logging.getLogger(__name__)

class LoR_VP(nn.Module):
    def __init__(self, cfg):
        super(LoR_VP, self).__init__()
        width = cfg.VP.bar_width
        height = cfg.VP.bar_height

        init_methods = cfg.VP.init_method.split(',')
        self.left_bar = torch.nn.Parameter(torch.empty(3, height, width))
        self.get_init(init_methods[0], self.left_bar)
        self.right_bar = torch.nn.Parameter(torch.empty(3, width, height))
        self.get_init(init_methods[1], self.right_bar)
        self.program = torch.bmm(self.left_bar, self.right_bar)


    def get_init(self, init_method, params):
        if init_method == 'zero':
            params.data.fill_(0)
        elif init_method == 'random':
            params.data.normal_(0, 1)
        elif init_method == 'xavier':
            torch.nn.init.xavier_uniform_(params)
        elif init_method == 'kaiming':
            torch.nn.init.kaiming_uniform_(params, nonlinearity='relu')
        elif init_method == 'uniform':
            torch.nn.init.uniform_(params, a=-0.1, b=0.1)
        elif init_method == 'normal':
            torch.nn.init.normal_(params, mean=0.0, std=0.01)

    def forward(self, x):
        self.program = torch.bmm(self.left_bar, self.right_bar).cuda()
        x = x + self.program
        return x