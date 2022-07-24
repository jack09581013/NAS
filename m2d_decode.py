import os
import sys
import numpy as np
import torch
from Map2D.decoding_formulas import Decoder
from config import *

class Loader(object):
    def __init__(self, config):
        self.config = config

        # Resuming checkpoint
        assert config.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(config.resume))
        assert os.path.isfile(config.resume), RuntimeError("=> no checkpoint found at '{}'".format(config.resume))

        checkpoint = torch.load(config.resume)

        self._alphas_fea = checkpoint['auto2d.alphas']
        self._betas_fea  = checkpoint['auto2d.betas']
        self.decoder_fea = Decoder(alphas=self._alphas_fea, betas=self._betas_fea, steps=self.config.step)

    def retreive_alphas_betas(self):
        return self._alphas_fea, self._betas_fea

    def decode_architecture(self):
        fea_paths, fea_paths_space = self.decoder_fea.viterbi_decode()
        return fea_paths, fea_paths_space

    def decode_cell(self):
        fea_genotype = self.decoder_fea.genotype_decode()
        return fea_genotype

def get_new_network_cell():
    config = Config_Map2D_Decode()
    load_model = Loader(config)
    auto2d_net_paths, auto2d_net_paths_space = load_model.decode_architecture()
    auto2d_genotype = load_model.decode_cell()
    print('Auto2D Net search results:', auto2d_net_paths)
    print('Auto2D Net cell structure:\n', auto2d_genotype)
    dir_name = os.path.dirname(config.resume)
    fea_net_path_filename = os.path.join(dir_name, 'auto2d_network_path')
    fea_genotype_filename = os.path.join(dir_name, 'auto2d_genotype')
    np.save(fea_net_path_filename, auto2d_net_paths)
    np.save(fea_genotype_filename, auto2d_genotype)

if __name__ == '__main__':
    get_new_network_cell()
