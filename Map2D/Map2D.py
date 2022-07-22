import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Map2D.decoding_formulas import network_layer_to_space
from Map2D.new_model_2d import newAuto2D
# from Map2D.skip_model_2d import newAuto2D


class Map2D(nn.Module):
    def __init__(self, config):
        super(Map2D, self).__init__()

        network_path_auto2d, cell_arch_auto2d = np.load(config.net_arch_auto2d), np.load(config.cell_arch_auto2d)
        print(f'Auto2D network path: {network_path_auto2d}\n')

        network_arch_auto2d = network_layer_to_space(network_path_auto2d)

        self.auto2d = newAuto2D(network_arch_auto2d, cell_arch_auto2d, config=config)        
        self.last_1 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.last_2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.last_3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)


    def forward(self, x):
        x = self.auto2d(x)
        x = self.last_1(x)
        x = self.last_2(x)
        x = self.last_3(x)
        return x

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params
