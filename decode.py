import os
import sys
import numpy as np
import torch
from LEAStereo.decoding_formulas import Decoder
from config import *

class Loader(object):
    def __init__(self, config):
        self.config = config

        # Resuming checkpoint
        assert config.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(config.resume))
        assert os.path.isfile(config.resume), RuntimeError("=> no checkpoint found at '{}'".format(config.resume))

        checkpoint = torch.load(config.resume)

        self._alphas_fea = checkpoint['feature.alphas']
        self._betas_fea  = checkpoint['feature.betas']
        self.decoder_fea = Decoder(alphas=self._alphas_fea, betas=self._betas_fea, steps=self.config.step)

        self._alphas_mat = checkpoint['matching.alphas']
        self._betas_mat  = checkpoint['matching.betas']
        self.decoder_mat = Decoder(alphas=self._alphas_mat, betas=self._betas_mat, steps=self.config.step)

    def retreive_alphas_betas(self):
        return self._alphas_fea, self._betas_fea, self._alphas_mat, self._betas_mat

    def decode_architecture(self):
        fea_paths, fea_paths_space = self.decoder_fea.viterbi_decode()
        mat_paths, mat_paths_space = self.decoder_mat.viterbi_decode()
        return fea_paths, fea_paths_space, mat_paths, mat_paths_space

    def decode_cell(self):
        fea_genotype = self.decoder_fea.genotype_decode()
        mat_genotype = self.decoder_mat.genotype_decode()
        return fea_genotype, mat_genotype

def get_new_network_cell():
    config = Config_Decode()
    load_model = Loader(config)
    fea_net_paths, fea_net_paths_space, mat_net_paths, mat_net_paths_space = load_model.decode_architecture()
    fea_genotype, mat_genotype = load_model.decode_cell()
    print('Feature Net search results:', fea_net_paths)
    print('Matching Net search results:', mat_net_paths)
    print('Feature Net cell structure:', fea_genotype)
    print('Matching Net cell structure:', mat_genotype)

    dir_name = os.path.dirname(config.resume)
    fea_net_path_filename = os.path.join(dir_name, 'feature_network_path')
    fea_genotype_filename = os.path.join(dir_name, 'feature_genotype')
    np.save(fea_net_path_filename, fea_net_paths)
    np.save(fea_genotype_filename, fea_genotype)

    mat_net_path_filename = os.path.join(dir_name, 'matching_network_path')
    mat_genotype_filename = os.path.join(dir_name, 'matching_genotype')
    np.save(mat_net_path_filename, mat_net_paths)
    np.save(mat_genotype_filename, mat_genotype)

    fea_cell_name = os.path.join(dir_name, 'feature_cell_structure')    
    mat_cell_name = os.path.join(dir_name, 'matching_cell_structure')

if __name__ == '__main__':
    get_new_network_cell()
