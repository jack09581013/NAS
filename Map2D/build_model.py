import torch
import torch.nn as nn
import torch.nn.functional as F
from Map2D.build_model_2d import Auto2D
import pdb
from time import time


class AutoMap2D(nn.Module):
    def __init__(self, layers=6, filter=8, block=4, step=3, device='cpu'):
        super(AutoMap2D, self).__init__()
        # define Feature parameters
        self.layers = layers
        self.filter = filter
        self.block = block
        self.step = step
        self.device = device

        self.auto2d = Auto2D(self.layers, self.filter, self.block, self.step)

    def forward(self, x):
        x = self.auto2d(x)
        return x

