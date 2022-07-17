from torch.utils.data import Dataset, Subset
import cv2
import numpy as np
import torch
import os
import utils


class FlyingThings3D(Dataset):

    # height, width = 540, 960
    def __init__(self, max_disparity, dataset_dir, type='train', image='cleanpass', use_crop_size=False, crop_size=None,
                 crop_seed=None, use_resize=False, resize=(None, None),
                 use_padding_crop_size=False, padding_crop_size=(None, None)):

        self.dataset_dir = dataset_dir
        assert os.path.exists(self.dataset_dir), 'Dataset path is not exist'
        self.data_max_disparity = []
        self.image = image
        self.use_crop_size = use_crop_size
        self.crop_size = crop_size
        self.crop_seed = crop_seed
        self.use_resize = use_resize
        self.resize = resize
        self.use_padding_crop_size = use_padding_crop_size
        self.padding_crop_size = padding_crop_size
        self.pass_info = {}

        if type == 'train':
            self.data_max_disparity.append(utils.load(os.path.join(self.dataset_dir, f'left_max_disparity.np'))[0])
            self.root = os.path.join(self.dataset_dir, 'TRAIN')
            self.size = 7460

        elif type == 'test':
            self.data_max_disparity.append(utils.load(os.path.join(self.dataset_dir, f'left_max_disparity.np'))[1])
            self.root = os.path.join(self.dataset_dir, 'TEST')
            self.size = 1440

        else:
            raise Exception(f'Unknown type: "{type}"')

        self.mask = np.ones(self.size, dtype=np.uint8)
        for d in self.data_max_disparity:
            self.mask = self.mask & (d < max_disparity - 1)
        self.size = np.sum(self.mask)
        self.mask = torch.from_numpy(self.mask)

        if image not in ['cleanpass', 'finalpass']:
            raise Exception(f'Unknown image: "{image}"')

        self._make_mask_index()

    def __getitem__(self, index):
        if self.use_crop_size:
            index = self.mask_index[index]
            X = utils.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))  # channel, height, width
            X = torch.from_numpy(X)

            cropper = utils.RandomCropper(X.shape[1:3], self.crop_size, seed=self.crop_seed)
            X = cropper.crop(X)
            X = X.float() / 255

            Y_list = []
            Y = utils.load(os.path.join(self.root, f'left_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            Y = cropper.crop(Y)
            Y_list.append(Y.unsqueeze(0))
            Y = torch.cat(Y_list, dim=0)

        elif self.use_resize:
            index = self.mask_index[index]
            X = utils.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))  # channel, height, width
            self.pass_info['original_height'], self.pass_info['original_width'] = X.shape[1:]

            X1 = X[:3, :, :].swapaxes(0, 2).swapaxes(0, 1)
            X2 = X[3:, :, :].swapaxes(0, 2).swapaxes(0, 1)

            X1 = cv2.resize(X1, (self.resize[1], self.resize[0]))
            X2 = cv2.resize(X2, (self.resize[1], self.resize[0]))

            X = np.concatenate([X1, X2], axis=2)
            X = X.swapaxes(0, 1).swapaxes(0, 2)
            X = torch.from_numpy(X).float() / 255.0

            Y_list = []
            Y = utils.load(os.path.join(self.root, f'left_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            Y_list.append(Y.unsqueeze(0))
            Y = torch.cat(Y_list, dim=0)

        elif self.use_padding_crop_size:
            index = self.mask_index[index]
            X = utils.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))  # channel, height, width

            self.pass_info['original_height'], self.pass_info['original_width'] = X.shape[1:]
            assert self.pass_info['original_height'] <= self.padding_crop_size[0]
            assert self.pass_info['original_width'] <= self.padding_crop_size[1]

            X_pad = np.zeros((6, *self.padding_crop_size), dtype=np.uint8)
            X_pad[:X.shape[0], :X.shape[1], :] = X[...]
            X = X_pad
            X = torch.from_numpy(X).float() / 255.0

            Y_list = []
            Y = utils.load(os.path.join(self.root, f'left_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            Y_list.append(Y.unsqueeze(0))
            Y = torch.cat(Y_list, dim=0)

        return X, Y, self.pass_info

    def __len__(self):
        return self.size

    def _make_mask_index(self):
        self.mask_index = np.zeros(self.size, dtype=np.int)

        i = 0
        m = 0
        while i < len(self.mask):
            if self.mask[i]:
                self.mask_index[m] = i
                m += 1
            i += 1

    def __str__(self):
        return 'FlyingThings3D'


class Map2D_Dataset(Dataset):

    # height, width = 540, 960
    def __init__(self, type, height, width):

        if type == 'trainA':
            torch.random.manual_seed(0)
            self.X = torch.rand(10, 4, height, width)
            self.Y = torch.rand(10, 3, height * 2, width * 2)
            self.size = 10

        elif type == 'trainB':
            torch.random.manual_seed(1)
            self.X = torch.rand(10, 4, height, width)
            self.Y = torch.rand(10, 3, height * 2, width * 2)
            self.size = 10

        elif type == 'test':
            torch.random.manual_seed(2)
            self.X = torch.rand(10, 4, height, width)
            self.Y = torch.rand(10, 3, height * 2, width * 2)
            self.size = 10

        else:
            raise Exception(f'Unknown type: "{type}"')

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.size

    def __str__(self):
        return 'Map2D_Dataset'


def random_subset(dataset, size, seed=None):
    assert size <= len(dataset), 'subset size cannot larger than dataset'
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    indexes = indexes[:size]
    return Subset(dataset, indexes)


def random_split(dataset, train_ratio=0.8, seed=None):
    assert 0 <= train_ratio <= 1
    train_size = int(train_ratio * len(dataset))
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    train_indexes = indexes[:train_size]
    test_indexes = indexes[train_size:]
    return Subset(dataset, train_indexes), Subset(dataset, test_indexes)


def sub_sampling(X, Y, ratio):
    X = X[:, ::ratio, ::ratio]
    Y = Y[::ratio, ::ratio] / ratio
    return X, Y
