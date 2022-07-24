class Config:
    def __init__(self):
        self.dataset_dir = None
        self.dataset_name = None
        self.version = None
        self.max_version = 2000  # KITTI 2015 v1497 recommended version
        self.batch = 1
        self.learning_rate = 0.001
        self.seed = 0
        self.is_debug = True
        self.num_workers = 0
        self.height = None
        self.width = None
        self.max_disparity = None
        self.device = 'cuda'

class Config_Flyingthings3D(Config):
    def __init__(self):
        super().__init__()
        self.dataset_dir = r'D:\Datasets\flyingthings3d_preprocessing'
        self.dataset_name = 'flyingthings3D'

        self.height = 288
        self.width = 576
        self.max_disparity = 96

class Config_Map2D(Config):
    def __init__(self):
        super().__init__()

        # self.height = 112
        # self.width = 112

        self.height = 16
        self.width = 16


class Config_LEAStereo_Search(Config_Flyingthings3D):
    def __init__(self):
        super().__init__()
        # AutoStereo settings
        self.fea_num_layers = 6
        self.fea_filter_multiplier = 4
        self.fea_block_multiplier = 3
        self.fea_step = 3
        self.mat_num_layers = 12
        self.mat_filter_multiplier = 4
        self.mat_block_multiplier = 3
        self.mat_step = 3

        # Other
        self.alpha_epoch = 3
        self.epoch = 10
        self.save_history_file_path = './images/history_AutoStereo.png'
        self.save_best_model_path = './models/retrain/LEAStereo/best_AutoStereo.pth'

        # Input size
        self.height = 72
        self.width = 96
        self.max_disparity = 96

        # self.height = 192
        # self.width = 384
        # self.max_disparity = 192

class Config_LEAStereo_Decode(Config_Flyingthings3D):
    def __init__(self):
        super().__init__()

        # decode
        self.resume = './models/retrain/LEAStereo/best_AutoStereo.pth'
        self.step = 3

class Config_LEAStereo_Train(Config_Flyingthings3D):
    def __init__(self):
        super().__init__()
        self.epoch = 1

        # LEAStereo settings
        self.fea_num_layers = 6
        self.fea_filter_multiplier = 8
        self.fea_block_multiplier = 4
        self.fea_step = 3
        self.mat_num_layers = 12
        self.mat_filter_multiplier = 8
        self.mat_block_multiplier = 4
        self.mat_step = 3

        # train
        self.resume = './models/release/best_LEAStereo.pth'
        self.net_arch_fea = './models/release/feature_network_path.npy'
        self.cell_arch_fea = './models/release/feature_genotype.npy'
        self.net_arch_mat = './models/release/matching_network_path.npy'
        self.cell_arch_mat = './models/release/matching_genotype.npy'

        self.save_history_file_path = './images/history_LEAStereo.png'
        self.save_best_model_path = './models/release/best_LEAStereo.pth'


class Config_Map2D_Search(Config_Map2D):
    def __init__(self):
        super().__init__()
        # settings 1
        # self.num_layers = 8
        # self.filter_multiplier = 8
        # self.block_multiplier = 4
        # self.step = 3

        # settings 2
        # self.num_layers = 4
        # self.filter_multiplier = 4
        # self.block_multiplier = 3
        # self.step = 3

        # settings 3
        self.num_layers = 8
        self.filter_multiplier = 4
        self.block_multiplier = 5
        self.step = 5

        # Other
        self.alpha_epoch = 3
        self.epoch = 40
        self.save_history_file_path = './images/history_AutoMap2D.png'
        self.save_best_model_path = './models/retrain/Map2D/best_AutoMap2D.pth'

class Config_Map2D_Decode(Config_Map2D):
    def __init__(self):
        super().__init__()

        # decode
        self.resume = './models/retrain/Map2D/best_AutoMap2D.pth'
        self.step = 5

class Config_Map2D_Train(Config_Map2D):
    def __init__(self):
        super().__init__()
        self.epoch = 50

        # settings 3
        self.num_layers = 8
        self.filter_multiplier = 4
        self.block_multiplier = 5
        self.step = 5

        # train
        self.resume = './models/retrain/Map2D/best_Map2D.pth'
        self.net_arch_auto2d = './models/retrain/Map2D/auto2d_network_path.npy'
        self.cell_arch_auto2d = './models/retrain/Map2D/auto2d_genotype.npy'

        self.save_history_file_path = './images/history_Map2D.png'
        self.save_best_model_path = './models/retrain/Map2D/best_Map2D.pth'


class Config_SimpleModel(Config):
    def __init__(self):
        super().__init__()
        self.epoch = 50
        self.resume = './models/retrain/SimpleModel/best_SimpleModel.pth'
        self.save_history_file_path = './images/history_SimpleModel.png'

        self.height = 16
        self.width = 16

