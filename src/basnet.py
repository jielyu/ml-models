# encoding: utf-8

## code from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import time
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import transform, color
from common_utils.check_utils import check_model_path

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'ResNet34P','ResNet50S','ResNet50P','ResNet101P']
#
# resnet18_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet18-5c106cde.pth'
# resnet34_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet34-333f7ec4.pth'
# resnet50_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet50-19c8e357.pth'
# resnet101_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet101-5d3b4d8f.pth'
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockDe(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockDe, self).__init__()

        self.convRes = conv3x3(inplanes, planes, stride)
        self.bnRes = nn.BatchNorm2d(planes)
        self.reluRes = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.convRes(x)
        residual = self.bnRes(residual)
        residual = self.reluRes(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


class BASNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BASNet, self).__init__()

        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7

        ## -------------Bridge--------------

        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  ###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode="bilinear")  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.upscore4 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.upscore3 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1, 64)

    def forward(self, x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32

        hx = self.pool4(h4)  # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5)  # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))  # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6)  # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5)  # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)  # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)  # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db)  # 8->256

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6)  # 8->256

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        ## -------------Refine Module-------------
        dout = self.refunet(d1)  # 256

        return (
            F.sigmoid(dout),
            F.sigmoid(d1),
            F.sigmoid(d2),
            F.sigmoid(d3),
            F.sigmoid(d4),
            F.sigmoid(d5),
            F.sigmoid(d6),
            F.sigmoid(db),
        )


class RescaleT(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(
            image, (self.output_size, self.output_size), mode="constant"
        )
        lbl = transform.resize(
            label,
            (self.output_size, self.output_size),
            mode="constant",
            order=0,
            preserve_range=True,
        )

        return {"image": img, "label": lbl}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        image, label = sample["image"], sample["label"]

        tmpLbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0])
            )
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1])
            )
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2])
            )
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0])
            )
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1])
            )
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2])
            )

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(
                tmpImg[:, :, 0]
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(
                tmpImg[:, :, 1]
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(
                tmpImg[:, :, 2]
            )
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(
                tmpImg[:, :, 3]
            )
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(
                tmpImg[:, :, 4]
            )
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(
                tmpImg[:, :, 5]
            )

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0])
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1])
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2])
            )

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(
                tmpImg[:, :, 0]
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(
                tmpImg[:, :, 1]
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(
                tmpImg[:, :, 2]
            )

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {"image": torch.from_numpy(tmpImg), "label": torch.from_numpy(tmpLbl)}


class BASNetWrapper:
    def __init__(self, model_path="models/ml4a_basnet/basnet.pth") -> None:
        check_model_path(model_path)
        self.model_path = model_path
        self.model = BASNet(3, 1)
        self.model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    @staticmethod
    def normPRED(d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def __call__(self, image):
        label = np.zeros(image.shape)
        sample = {"image": image, "label": label}
        size = image.shape[1], image.shape[0]
        # 预处理样本数据
        transform = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])
        sample = transform(sample)
        # 转换输入数据的格式
        inputs_test = sample["image"]
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        inputs_test = inputs_test.unsqueeze(0)
        # 运行模型
        outputs_test = self.model(inputs_test)
        pred = outputs_test[0][:, 0, :, :]
        pred = self.normPRED(pred)
        # 转换输出数据的格式
        pred = pred.squeeze()
        pred_np = pred.cpu().data.numpy()
        im_mask = Image.fromarray(pred_np * 255).convert("RGB")
        im_mask = im_mask.resize(size, resample=Image.BILINEAR)
        im_mask = np.array(im_mask)
        return im_mask

    @staticmethod
    def mask_image(img, mask):
        img, mask = np.array(img), np.array(mask) / 255.0
        mask_img = img * mask
        return mask_img.astype(img.dtype)

    def demo(self, axis=False):
        img_path = "data/basnet-samples/Mona_Lisa.jpg"
        ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        factor = 3
        uni_size = (int(220 * factor), int(350 * factor))
        resized_img = cv2.resize(img, uni_size)
        start_time = time.time()
        mask = self(resized_img)
        end_time = time.time()
        print("cost time(s):", end_time - start_time)
        # 计算mask后的图像
        mask_img = self.mask_image(resized_img, mask)

        # 可视化
        plt.figure()
        ax1 = plt.subplot(1, 3, 1)
        plt.imshow(resized_img)
        plt.title("origin image")
        ax2 = plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.title("mask")
        ax3 = plt.subplot(1, 3, 3)
        plt.imshow(mask_img)
        plt.title("masked image")
        if axis is False:
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
        plt.savefig("output/basnet_foreground.png", bbox_inches="tight", dpi=300)

        # 处理背景
        bg_img = cv2.imread("data/basnet-samples/starnight.jpg", cv2.IMREAD_UNCHANGED)
        resized_bg_img = cv2.resize(bg_img, uni_size)
        bg_mask = 255 - mask
        mask_bg_img = self.mask_image(resized_bg_img, bg_mask)

        # 可视化
        plt.figure()
        ax1 = plt.subplot(1, 3, 1)
        plt.imshow(resized_bg_img)
        plt.title("origin image")
        ax2 = plt.subplot(1, 3, 2)
        plt.imshow(bg_mask)
        plt.title("mask")
        ax3 = plt.subplot(1, 3, 3)
        plt.imshow(mask_bg_img)
        plt.title("masked image")
        if axis is False:
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
        plt.savefig("output/basnet_background.png", bbox_inches="tight", dpi=300)

        final_img = mask_img + mask_bg_img
        plt.figure()
        plt.imshow(final_img)
        plt.title("final image")
        if axis is False:
            plt.axis("off")
        plt.savefig("output/basnet_final.png", bbox_inches="tight", dpi=300)

        plt.show()


def main():
    basnet = BASNetWrapper()
    basnet.demo()


if __name__ == "__main__":
    main()
