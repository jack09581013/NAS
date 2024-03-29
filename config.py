import os


class Config:
    def __init__(self):
        self.dataset_dir = None
        self.dataset_name = None
        self.version = None
        self.max_version = 1000  # KITTI 2015 v1497 recommended version
        self.batch = 1
        self.learning_rate = 0.001
        self.seed = 0
        self.is_debug = True
        self.num_workers = 0
        self.height = None
        self.width = None
        self.device = 'cuda'

class Config_Flyingthings3D(Config):
    def __init__(self):
        super().__init__()
        self.dataset_dir = 'D:/Datasets/flyingthings3d_preprocessing'
        self.dataset_name = 'flyingthings3D'

        self.height = 288
        self.width = 576
        self.max_disparity = 96

class Config_Map2D(Config):
    def __init__(self):
        super().__init__()
        self.root = './models/retrain/Map2D/'
        os.makedirs(self.root, exist_ok=True)
        os.makedirs('./images', exist_ok=True)
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
        # settings: original segmtation paper
        # self.num_layers = 12
        # self.filter_multiplier = 8
        # self.block_multiplier = 5
        # self.step = 5

        # settings: LEAStereo train
        # self.num_layers = 8
        # self.filter_multiplier = 8
        # self.block_multiplier = 4
        # self.step = 3

        # settings LEAStereo train (less feature)
        # self.num_layers = 8
        # self.filter_multiplier = 4
        # self.block_multiplier = 4
        # self.step = 3

        # settings LEAStereo search
        # self.num_layers = 8
        # self.filter_multiplier = 4
        # self.block_multiplier = 3
        # self.step = 3

        # settings: test version
        # self.num_layers = 8
        # self.filter_multiplier = 4
        # self.block_multiplier = 5
        # self.step = 5

        # settings: test version
        self.num_layers = 4
        self.filter_multiplier = 2
        self.block_multiplier = 2
        self.step = 2

        # Other
        self.alpha_epoch = 3
        self.epoch = 40
        self.save_history_file_path = './images/history_AutoMap2D.png'
        self.resume = os.path.join(self.root, 'best_AutoMap2D.pth')

class Config_Map2D_Decode(Config_Map2D):
    def __init__(self):
        super().__init__()
        search_config = Config_Map2D_Search()

        # decode
        self.resume = search_config.resume
        self.step = search_config.step

class Config_Map2D_Train(Config_Map2D):
    def __init__(self):
        super().__init__()
        self.epoch = 50
        search_config = Config_Map2D_Search()

        # settings 3
        self.num_layers = search_config.num_layers
        self.filter_multiplier = search_config.filter_multiplier
        self.block_multiplier = search_config.block_multiplier
        self.step = search_config.step

        # train
        self.model_path = os.path.join(self.root, 'best_Map2D.pth')
        self.net_arch_auto2d = os.path.join(self.root, 'auto2d_network_path.npy')
        self.cell_arch_auto2d = os.path.join(self.root, 'auto2d_genotype.npy')
        self.save_history_file_path = './images/history_Map2D.png'


class Config_SimpleModel(Config):
    def __init__(self):
        super().__init__()
        self.epoch = 50
        self.resume = './models/retrain/SimpleModel/best_SimpleModel.pth'
        self.save_history_file_path = './images/history_SimpleModel.png'

        # Map2D
        # self.height = 16
        # self.width = 16

        # EfficientNet_V2
        self.height = 64
        self.width = 64

class Config_EfficientNet_V2(Config):
    def __init__(self):
        super().__init__()

        self.height = 64
        self.width = 64

        self.epoch = 50

        # train
        self.model_path = './models/retrain/EfficientNet_V2/best_EfficientNet_V2.pth'
        self.save_history_file_path = './images/history_EfficientNet_V2.png'

