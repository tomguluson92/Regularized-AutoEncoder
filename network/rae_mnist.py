# -*- coding: utf-8 -*-

"""
    Regularized auto-encoder, RAE, ICLR2020.
    @author: samuel ko
    @date:   2020.05.06

"""

import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.nn import ModuleList

import torch
from torch import nn

import os
import cv2
from skimage.io import imread
from opts.opts import TrainOptions, INFO

import copy
from tqdm import tqdm

from torchvision.utils import save_image


# =========================================================================
#   Define Encoder
# =========================================================================

class Encoder(nn.Module):
    def __init__(self,
                 num_filters=128,
                 bottleneck_size=16,
                 include_batch_norm=True):

        super(Encoder, self).__init__()

        self.include_bn = include_batch_norm

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=num_filters,
                               kernel_size=4,
                               stride=2,
                               padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(in_channels=num_filters,
                               out_channels=num_filters * 2,
                               kernel_size=4,
                               stride=2,
                               padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(num_filters * 2)

        self.conv3 = nn.Conv2d(in_channels=num_filters * 2,
                               out_channels=num_filters * 4,
                               kernel_size=4,
                               stride=2,
                               padding=(2, 2))
        self.bn3 = nn.BatchNorm2d(num_filters * 4)

        self.conv4 = nn.Conv2d(in_channels=num_filters * 4,
                               out_channels=num_filters * 8,
                               kernel_size=4,
                               stride=2,
                               padding=(2, 2))
        self.bn4 = nn.BatchNorm2d(num_filters * 8)

        self.fc = nn.Linear(num_filters * 72, bottleneck_size)

    def forward(self, x):

        x = self.conv1(x)
        if self.include_bn:
            x = self.bn1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv2(x)
        if self.include_bn:
            x = self.bn2(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv3(x)
        if self.include_bn:
            x = self.bn3(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv4(x)
        if self.include_bn:
            x = self.bn4(x)
        x = F.leaky_relu(x, 0.1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# =========================================================================
#   Define Decoder
# =========================================================================

class Decoder(nn.Module):
    def __init__(self,
                 num_filters=128,
                 bottleneck_size=16,
                 include_batch_norm=True):

        super(Decoder, self).__init__()

        self.include_bn = include_batch_norm

        self.conv1 = nn.ConvTranspose2d(in_channels=1024,
                                        out_channels=num_filters * 4,
                                        kernel_size=4,
                                        stride=2,
                                        padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(num_filters * 4)

        self.conv2 = nn.ConvTranspose2d(in_channels=num_filters * 4,
                                        out_channels=num_filters * 2,
                                        kernel_size=4,
                                        stride=2,
                                        padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(num_filters * 2)

        self.conv3 = nn.ConvTranspose2d(in_channels=num_filters * 2,
                                        out_channels=1,
                                        kernel_size=5,
                                        padding=(1, 1))
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(bottleneck_size, 8 * 8 * 1024)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)

        x = self.conv1(x)
        if self.include_bn:
            x = self.bn1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv2(x)
        if self.include_bn:
            x = self.bn2(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv3(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    a = torch.randn(1, 1, 28, 28)
    enc = Encoder()
    dec = Decoder()
    print(enc(a).shape)
    # print(dec(enc(a)).shape)
