import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_size = 4
        for i in range(self.layer_size):
            self.layers.append(ConvBR(4, 4))

        self.last_1 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.last_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.last_3 = nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        for i in range(self.layer_size):
            x = self.layers[i](x)

        x = self.last_1(x)
        x = self.last_2(x)
        x = self.last_3(x)

        return x

class ConvBR(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, bn=True, relu=True):
        super(ConvBR, self).__init__()
        self.use_bn = bn
        self.use_relu = relu
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x
