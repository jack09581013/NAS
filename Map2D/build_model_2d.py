import torch.nn as nn
import torch.nn.functional as F
import Map2D.cell_level_search_2d as cell_level_search
from Map2D.genotypes_2d import PRIMITIVES
from Map2D.operations_2d import *
from Map2D.decoding_formulas import Decoder
import pdb


class Auto2D(nn.Module):
    def __init__(self, num_layers, filter_multiplier=8, block_multiplier=4, step=4, cell=cell_level_search.Cell):
        super(Auto2D, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._initialize_alphas_betas()
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = f_initial * self._block_multiplier

        print(f'Feature Net block_multiplier: {block_multiplier}')
        print(f'Feature Net filter_multiplier: {filter_multiplier}')
        print(f'Feature Net f_initial: {f_initial}')

        self.stem0 = ConvBR(4, half_f_initial * self._block_multiplier, 3, stride=1, padding=1)
        self.stem1 = ConvBR(half_f_initial * self._block_multiplier, half_f_initial * self._block_multiplier, 3, stride=1, padding=1)
        # self.stem2 = ConvBR(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, 3, stride=1, padding=1)
        self.stem2 = nn.ConvTranspose2d(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, kernel_size=4, stride=2, padding=1, bias=False) 

        for i in range(self._num_layers):
            if i == 0:
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]
        
        self.upsample_4 = nn.ConvTranspose2d(self._num_end, self._num_end, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(self._num_end * 2, self._num_end * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_16 = nn.ConvTranspose2d(self._num_end * 4, self._num_end * 4, kernel_size=4, stride=2, padding=1, bias=False)

        self.last_2 = ConvBR(self._num_end, 3, 1, 1, 0, bn=False, relu=False)
        self.last_4 = ConvBR(self._num_end * 2, self._num_end, 1, 1, 0, bn=False, relu=False)
        self.last_8 = ConvBR(self._num_end * 4, self._num_end * 2, 1, 1, 0, bn=False, relu=False)
        self.last_16 = ConvBR(self._num_end * 8, self._num_end * 4, 1, 1, 0, bn=False, relu=False)

    def forward(self, x):
        self.level_2 = []
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []

        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)

        self.level_2.append(stem2)

        count = 0
        normalized_betas = torch.randn(self._num_layers, 4, 3)
        # Softmax on alphas and betas
        normalized_alphas = F.softmax(self.alphas, dim=-1)

        for layer in range(len(self.betas)):
            if layer == 0:
                normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)

            elif layer == 1:
                normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)
                normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)

            elif layer == 2:
                normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)
                normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
            else:
                normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2 / 3)
                normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1) * (2 / 3)

        for layer in range(self._num_layers):

            if layer == 0:
                level2_new, = self.cells[count](None, None, self.level_2[-1], None, normalized_alphas)
                count += 1
                level4_new, = self.cells[count](None, self.level_2[-1], None, None, normalized_alphas)
                count += 1

                level2_new = normalized_betas[layer][0][1] * level2_new
                level4_new = normalized_betas[layer][0][2] * level4_new
                self.level_2.append(level2_new)
                self.level_4.append(level4_new)

            elif layer == 1:
                level2_new_1, level2_new_2 = self.cells[count](self.level_2[-2],
                                                               None,
                                                               self.level_2[-1],
                                                               self.level_4[-1],
                                                               normalized_alphas)
                count += 1
                level2_new = normalized_betas[layer][0][1] * level2_new_1 + normalized_betas[layer][1][0] * level2_new_2

                level4_new_1, level4_new_2 = self.cells[count](None,
                                                               self.level_2[-1],
                                                               self.level_4[-1],
                                                               None,
                                                               normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][2] * level4_new_1 + normalized_betas[layer][1][2] * level4_new_2

                level8_new, = self.cells[count](None,
                                                self.level_4[-1],
                                                None,
                                                None,
                                                normalized_alphas)
                level8_new = normalized_betas[layer][1][2] * level8_new
                count += 1

                self.level_2.append(level2_new)
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)

            elif layer == 2:
                level2_new_1, level2_new_2 = self.cells[count](self.level_2[-2],
                                                               None,
                                                               self.level_2[-1],
                                                               self.level_4[-1],
                                                               normalized_alphas)
                count += 1
                level2_new = normalized_betas[layer][0][1] * level2_new_1 + normalized_betas[layer][1][0] * level2_new_2

                level4_new_1, level4_new_2, level4_new_3 = self.cells[count](self.level_4[-2],
                                                                             self.level_2[-1],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][2] * level4_new_1 + normalized_betas[layer][1][
                    1] * level4_new_2 + normalized_betas[layer][2][
                                 0] * level4_new_3

                level8_new_1, level8_new_2 = self.cells[count](None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               None,
                                                               normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][1][2] * level8_new_1 + normalized_betas[layer][2][
                    1] * level8_new_2

                level16_new, = self.cells[count](None,
                                                 self.level_8[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level16_new = normalized_betas[layer][2][2] * level16_new
                count += 1

                self.level_2.append(level2_new)
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)

            elif layer == 3:
                level2_new_1, level2_new_2 = self.cells[count](self.level_2[-2],
                                                               None,
                                                               self.level_2[-1],
                                                               self.level_4[-1],
                                                               normalized_alphas)
                count += 1
                level2_new = normalized_betas[layer][0][1] * level2_new_1 + normalized_betas[layer][1][0] * level2_new_2

                level4_new_1, level4_new_2, level4_new_3 = self.cells[count](self.level_4[-2],
                                                                             self.level_2[-1],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             normalized_alphas)
                count += 1
                level4_new = normalized_betas[layer][0][2] * level4_new_1 + normalized_betas[layer][1][
                    1] * level4_new_2 + normalized_betas[layer][2][
                                 0] * level4_new_3

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][1][2] * level8_new_1 + normalized_betas[layer][2][
                    1] * level8_new_2 + normalized_betas[layer][3][
                                 0] * level8_new_3

                level16_new_1, level16_new_2 = self.cells[count](None,
                                                                 self.level_8[-1],
                                                                 self.level_16[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][2][2] * level16_new_1 + normalized_betas[layer][3][
                    1] * level16_new_2

                self.level_2.append(level2_new)
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)

            else:
                level2_new_1, level2_new_2 = self.cells[count](self.level_2[-2],
                                                               None,
                                                               self.level_2[-1],
                                                               self.level_4[-1],
                                                               normalized_alphas)
                count += 1
                level2_new = normalized_betas[layer][0][1] * level2_new_1 + normalized_betas[layer][1][0] * level2_new_2

                level4_new_1, level4_new_2, level4_new_3 = self.cells[count](self.level_4[-2],
                                                                             self.level_2[-1],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             normalized_alphas)
                count += 1

                level4_new = normalized_betas[layer][0][2] * level4_new_1 + normalized_betas[layer][1][
                    1] * level4_new_2 + normalized_betas[layer][2][
                                 0] * level4_new_3

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas)
                count += 1
                level8_new = normalized_betas[layer][1][2] * level8_new_1 + normalized_betas[layer][2][
                    1] * level8_new_2 + normalized_betas[layer][3][
                                 0] * level8_new_3

                level16_new_1, level16_new_2 = self.cells[count](self.level_16[-2],
                                                                 self.level_8[-1],
                                                                 self.level_16[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level16_new = normalized_betas[layer][2][2] * level16_new_1 + normalized_betas[layer][3][
                    1] * level16_new_2

                self.level_2.append(level2_new)
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)

            self.level_2 = self.level_2[-2:]
            self.level_4 = self.level_4[-2:]
            self.level_8 = self.level_8[-2:]
            self.level_16 = self.level_16[-2:]

        # define upsampling
        result_2 = self.last_2(self.level_2[-1])
        result_4 = self.last_2(self.upsample_4(self.last_4(self.level_4[-1])))
        result_8 = self.last_2(self.upsample_4(self.last_4(self.upsample_8(self.last_8(self.level_8[-1])))))
        result_16 = self.last_2(self.upsample_4(self.last_4(self.upsample_8(self.last_8(self.upsample_16(self.last_16(self.level_16[-1])))))))

        sum_map = result_2 + result_4 + result_8 + result_16
        return sum_map

    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)

        self._arch_parameters = [
            alphas,
            betas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in
         zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()
