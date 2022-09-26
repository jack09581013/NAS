import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Map2D.genotypes_2d import PRIMITIVES
from Map2D.genotypes_2d import Genotype
from Map2D.operations_2d import *
import torch.nn.functional as F
import numpy as np
import pdb


class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ConvBR(self.C_prev_prev, self.C_out, 1, 1, 0)
        self.preprocess = ConvBR(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1)
            self._ops.append(op)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                               mode='bilinear', align_corners=True)

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)
        
        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
        return prev_input, concat_feature


class newAuto2D(nn.Module):
    def __init__(self, network_arch, cell_arch, cell=Cell, config=None):
        super(newAuto2D, self).__init__()
        self.config = config
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = config.step
        self._num_layers = config.num_layers
        self._block_multiplier = config.block_multiplier
        self._filter_multiplier = config.filter_multiplier

        initial_fm = self._filter_multiplier * self._block_multiplier
        half_initial_fm = initial_fm // 2

        self.stem0 = ConvBR(4, half_initial_fm, 3, stride=1, padding=1)
        self.stem1 = ConvBR(half_initial_fm, initial_fm, 3, stride=1, padding=1)
        # self.stem2 = ConvBR(initial_fm, initial_fm, 3, stride=1, padding=1)
        self.stem2 = nn.ConvTranspose2d(initial_fm, initial_fm, kernel_size=4, stride=2, padding=1, bias=False) 

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier,
                             self.cell_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 initial_fm / self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample)

            self.cells += [_cell]

        self.upsample_4 = nn.ConvTranspose2d(initial_fm, initial_fm, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(initial_fm * 2, initial_fm * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_16 = nn.ConvTranspose2d(initial_fm * 4, initial_fm * 4, kernel_size=4, stride=2, padding=1, bias=False)

        self.last_2 = ConvBR(initial_fm, 3, 1, 1, 0, bn=False, relu=False)
        self.last_4 = ConvBR(initial_fm * 2, initial_fm, 1, 1, 0, bn=False, relu=False)
        self.last_8 = ConvBR(initial_fm * 4, initial_fm * 2, 1, 1, 0, bn=False, relu=False)
        self.last_16 = ConvBR(initial_fm * 8, initial_fm * 4, 1, 1, 0, bn=False, relu=False)

    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)
        out = (stem1, stem2)

        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])

        last_output = out[-1]

        h, w = stem2.size()[2], stem2.size()[3]
        if last_output.size()[2] == h:
            fea = self.last_2(last_output)
        elif last_output.size()[2] == h // 2:
            fea = self.last_2(self.upsample_4(self.last_4(last_output)))
        elif last_output.size()[2] == h // 4:
            fea = self.last_2(self.upsample_4(self.last_4(self.upsample_8(self.last_8(last_output)))))
        elif last_output.size()[2] == h // 8:
            fea = self.last_2(self.upsample_4(self.last_4(self.upsample_8(self.last_8(self.upsample_16(self.last_16(last_output)))))))

        return fea

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params
