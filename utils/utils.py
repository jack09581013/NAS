import re
import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
import torch
import os
from colorama import Fore, Style
import sys
import pickle
import datetime
import math


def save(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


TOOLS_CURRENT_TIME = None


def tic():
    global TOOLS_CURRENT_TIME
    TOOLS_CURRENT_TIME = datetime.datetime.now()


def toc(return_timespan=False):
    global TOOLS_CURRENT_TIME
    if return_timespan:
        return datetime.datetime.now() - TOOLS_CURRENT_TIME
    else:
        print('Elapsed:', datetime.datetime.now() - TOOLS_CURRENT_TIME)


def get_latest_version(file_path):
    version_codes = [version_code(x) for x in os.listdir(file_path)]
    if len(version_codes) > 0:
        version_codes.sort()
        return version_codes[-1]
    else:
        return None


def version_code(file):
    a = file.index('-')
    b = file.index('.')
    return int(file[a + 1:b])


def timespan_str(timespan):
    total = timespan.seconds
    second = total % 60 + timespan.microseconds / 1e+06
    total //= 60
    minute = int(total % 60)
    total //= 60
    return f'{minute:02d}:{second:05.2f}'


def threshold_color(loss):
    if loss >= 10:
        return Fore.RED
    elif loss >= 5:
        return Fore.YELLOW
    elif loss >= 1:
        return Fore.BLUE
    elif loss >= 0.5:
        return Fore.CYAN
    else:
        return Fore.GREEN


def plot_image(img1, img2=None, title=None, target=False):
    if target:
        img1 = tensor_to_rgb_image(img1)
        if img2 is not None:
            img2 = tensor_to_target_image(img2)
    else:
        img1 = tensor_to_rgb_image(img1)
        if img2 is not None:
            img2 = tensor_to_rgb_image(img2)

    plt.figure(figsize=(19.2, 12))
    if img2 is None:
        if title is not None:
            plt.title(title)
        plt.imshow(img1)
    else:
        plt.subplot(121)
        if title is not None:
            plt.title(title)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
    plt.show()


def tensor_to_target_image(img):
    img = (img.data.cpu().numpy()).astype('uint8')
    img = np.squeeze(img, axis=0)
    return img


def tensor_to_rgb_image(img):
    img = (img.data.cpu().numpy() * 255).astype('uint8')
    img = img.transpose(1, 2, 0)
    new_image = np.zeros(img.shape, dtype='uint8')
    new_image[:, :, 0] = img[:, :, 2]
    new_image[:, :, 1] = img[:, :, 1]
    new_image[:, :, 2] = img[:, :, 0]
    img = new_image
    return img


def trend_regression(loss_trend, method='corr'):
    """Loss descent checking"""
    if method == 'regression':
        b = loss_trend.reshape(-1, 1)  # b: (n, 1)
        A = np.concatenate([np.arange(len(b)).reshape(-1, 1), np.ones((len(b), 1))], axis=1)
        x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        return float(x[0, 0])

    elif method == 'corr':
        A = loss_trend
        B = np.arange(len(A))
        corr = np.corrcoef(A, B)[0, 1]
        return corr

    else:
        raise Exception(f'Method "{method}" is not valid')


def read_pfm(file):
    # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)

        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False

        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)

        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = struct.unpack(fmt, buffer)

        img = np.array(img, dtype=np.float32).reshape(height, width, channels)
        img = np.flip(img, axis=0)
        # return img, height, width
        return img


class RandomCropper:
    def __init__(self, image_size, crop_size, seed=None):
        H, W = crop_size
        assert image_size[0] >= H, 'image height must larger than crop height'
        assert image_size[1] >= W, 'image width must larger than crop width'

        H_range = image_size[0] - H
        W_range = image_size[1] - W

        np.random.seed(seed)
        if H_range > 0:
            self.min_row = np.random.randint(0, H_range + 1)
        else:
            self.min_row = 0

        if W_range > 0:
            self.min_col = np.random.randint(0, W_range + 1)
        else:
            self.min_col = 0

        self.max_row = self.min_row + H
        self.max_col = self.min_col + W

    def crop(self, I):
        return I[..., self.min_row:self.max_row, self.min_col:self.max_col]