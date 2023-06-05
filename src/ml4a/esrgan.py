# encoding: utf-8

"""
参考代码： https://github.com/ml4a/ml4a
参考代码： https://github.com/xinntao/ESRGAN
"""

import sys

sys.path.append("src/")

import os
import time
import functools

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils.download_utils import download_from_gdrive


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        fea = self.lrelu(
            self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class ESRGan:
    @staticmethod
    def download_model():
        model_path = download_from_gdrive(
            "1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene",
            "ml4a_esrgan/RRDB_ESRGAN_x4.pth",
        )

    def __init__(self, model_path="models/ml4a_esrgan/RRDB_ESRGAN_x4.pth") -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def __call__(self, img):
        img = img * 1.0 / 255
        chw = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(chw).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        with torch.no_grad():
            output = (
                self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            )
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output


def main():
    # 创建模型
    model = ESRGan()
    # 读取图片
    img_path = "data/esrgan-samples/baboon.png"
    ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    # 生成超分辨率图片
    start_time = time.time()
    img_LR = model(img)
    end_time = time.time()
    print("Cost Time(s):", end_time - start_time)
    # 可视化
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original Image:{}".format((img.shape[:2])))
    plt.subplot(1, 2, 2)
    plt.imshow(img_LR)
    plt.axis("off")
    plt.title("Large Resolution 4x:{}".format((img_LR.shape[:2])))
    plt.savefig("output/esrgan.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    # ESRGan.download_model()
