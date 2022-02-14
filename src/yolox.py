#encoding: utf-8
"""
用于实现基于Pytorch框架的YOLOX模型的脚本

**说明:** 本实现参考或复用项目见github: https://github.com/Megvii-BaseDetection/YOLOX
"""
from __future__ import absolute_import
import enum

import os
import math
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import grid
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from loguru import logger


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction,
                num_classes,
                conf_thre=0.7,
                nms_thre=0.45,
                class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes],
                                           1,
                                           keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >=
                     conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(in_channels,
                              out_channels,
                              ksize=1,
                              stride=1,
                              groups=1,
                              act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels,
                               mid_channels,
                               ksize=1,
                               stride=1,
                               act="lrelu")
        self.layer2 = BaseConv(mid_channels,
                               in_channels,
                               ksize=3,
                               stride=1,
                               act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels,
                              out_channels,
                              1,
                              stride=1,
                              act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=act)
        self.conv2 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=act)
        self.conv3 = BaseConv(2 * hidden_channels,
                              out_channels,
                              1,
                              stride=1,
                              act=act)
        module_list = [
            Bottleneck(hidden_channels,
                       hidden_channels,
                       shortcut,
                       1.0,
                       depthwise,
                       act=act) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4,
                             out_channels,
                             ksize,
                             stride,
                             act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
            self,
            depth,
            in_channels=3,
            stem_out_channels=32,
            out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels,
                     stem_out_channels,
                     ksize=3,
                     stride=1,
                     act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in
        # python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2],
                                 in_channels * 2),
        )

    def make_group_layer(self,
                         in_channels: int,
                         num_blocks: int,
                         stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels,
                     in_channels * 2,
                     ksize=3,
                     stride=stride,
                     act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(*[
            BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
            BaseConv(
                filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            SPPBottleneck(
                in_channels=filters_list[1],
                out_channels=filters_list[0],
                activation="lrelu",
            ),
            BaseConv(
                filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            BaseConv(
                filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
        ])
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):

    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16,
                          base_channels * 16,
                          activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width),
                                      int(in_channels[1] * width),
                                      1,
                                      1,
                                      act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width),
                                     int(in_channels[0] * width),
                                     1,
                                     1,
                                     act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * width),
                             int(in_channels[0] * width),
                             3,
                             2,
                             act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(int(in_channels[1] * width),
                             int(in_channels[1] * width),
                             3,
                             2,
                             act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOXHead(nn.Module):

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        img_size=[640, 640],
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                ))
            self.cls_convs.append(
                nn.Sequential(*[
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                ]))
            self.reg_convs.append(
                nn.Sequential(*[
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                ]))
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))

        self.strides = strides
        self.img_size = img_size
        # 生成嵌入网格
        self.grids = []
        self.feat_sizes = []
        for _, stride in enumerate(self.strides):
            feat_h, feat_w = int(self.img_size[0] / stride), int(
                self.img_size[1] / stride)
            self.feat_sizes.append([feat_h, feat_w])
            yv, xv = torch.meshgrid(
                [torch.arange(feat_h),
                 torch.arange(feat_w)])
            grid = torch.stack((xv, yv), 2).view(1, 1, feat_h, feat_w,
                                                 2).type(torch.float32)
            self.grids.append(grid)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin):
        outputs = []
        origin_reg = []
        origin_obj = []
        origin_cls = []
        origin_grid = []
        expanded_strides = []

        for k, x in enumerate(xin):
            print("iter={}, x.shape={}".format(k, x.shape))
            stride_this_level = self.strides[k]
            # 分离不同任务的分支结构
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            # print("iter={}, cls_output.shape={}".format(k, cls_output.shape))
            # print("iter={}, reg_output.shape={}".format(k, reg_output.shape))
            # print("iter={}, obj_output.shape={}".format(k, obj_output.shape))

            # 进行网格嵌入的解码
            batch_size = x.shape[0]
            n_ch = 5 + self.num_classes
            output = torch.cat([
                reg_output,
                torch.sigmoid(obj_output),
                torch.sigmoid(cls_output)
            ], 1)
            hsize, wsize = output.shape[-2:]
            output = output.view(batch_size, self.n_anchors, n_ch, hsize,
                                 wsize)
            output = output.permute(0, 1, 3, 4, 2)
            output = output.reshape(batch_size, self.n_anchors * hsize * wsize,
                                    -1)
            grid = self.grids[k].view(1, -1, 2)
            output[..., :2] = (output[..., :2] + grid) * stride_this_level
            output[..., 2:4] = torch.exp(output[..., 2:4]) * stride_this_level
            outputs.append(output)

            # 保留原始输出bbox，便于损失函数的构建
            if self.training is True:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                # bbox回归
                reg_output = reg_output.view(batch_size, self.n_anchors, 4,
                                             hsize, wsize)
                reg_output = reg_output.permute(0, 1, 3, 4,
                                                2).reshape(batch_size, -1, 4)
                origin_reg.append(reg_output.clone())
                # 目标判断
                obj_output = obj_output.permute(0, 2, 3, 1).reshape(
                    batch_size, self.n_anchors * hsize * wsize, -1)
                origin_obj.append(obj_output)
                # 类别判定
                cls_output = cls_output.permute(0, 2, 3, 1).reshape(
                    batch_size, self.n_anchors * hsize * wsize, -1)
                origin_cls.append(cls_output)
                # 其他附带信息
                origin_grid.append(grid)
                expanded_strides.append(
                    torch.zeros(
                        1, grid.shape[1]).fill_(stride_this_level).type_as(x))
        pred_output = torch.cat(outputs, dim=1)
        return pred_output, origin_reg, origin_obj, origin_cls, origin_grid, expanded_strides


class IOUloss(nn.Module):

    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2),
                       (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2),
                       (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YoloxLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

    def forward(self, pred_output, origin_reg, origin_obj, origin_cls,
                origin_grid, expanded_strides, labels):
        assert len(origin_reg) == len(origin_obj) and len(origin_obj) == len(
            origin_cls) and len(origin_cls) == len(
                origin_grid) and len(origin_grid) > 0
        # 输入数据格式预处理
        bbox_preds = pred_output[:, :, :4]  # [batchsize, n_preds, 4]obj_preds
        obj_preds = torch.cat(origin_obj, dim=1)  # [batchsize, n_preds, 1]
        cls_preds = torch.cat(origin_cls, dim=1)  # [batchsize, n_preds, n_cls]
        # 网格坐标信息
        x_shifts, y_shifts = [], []
        for grid in origin_grid:
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_preds]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_preds]
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, n_preds]
        origin_reg = torch.cat(origin_reg, 1)
        num_classes = cls_preds.shape[-1]

        # 逐个样本建立top K匹配并完成label信息的嵌入
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # 目标个数
        total_num_anchors = pred_output.shape[1]  # 预测目标个数
        for batch_idx in range(pred_output.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = pred_output.new_zeros((0, num_classes))
                reg_target = pred_output.new_zeros((0, 4))
                l1_target = pred_output.new_zeros((0, 4))
                obj_target = pred_output.new_zeros((total_num_anchors, 1))
                fg_mask = pred_output.new_zeros(total_num_anchors).bool()
            else:
                # 建立匹配
                gt_bboxes_per_image = labels[batch_idx, :num_gt,
                                             1:5]  # [n_gt, 4]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [n_gt]
                bboxes_preds_per_image = bbox_preds[batch_idx]  # [n_preds, 4]
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(  # noqa
                    batch_idx,
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    obj_preds,
                )

                torch.cuda.empty_cache()
                num_fg += num_fg_img
                # 创建目标
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64),
                    num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                l1_target = self.get_l1_target(
                    pred_output.new_zeros((num_fg_img, 4)),
                    gt_bboxes_per_image[matched_gt_inds],
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                )
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(torch.float32))
            fg_masks.append(fg_mask)
            l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        l1_targets = torch.cat(l1_targets, 0)

        # 分别计算损失
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(
            bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1),
                                         obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(
            cls_preds.view(-1, num_classes)[fg_masks],
            cls_targets)).sum() / num_fg
        loss_l1 = 0.0
        if self.use_l1:
            loss_l1 = (self.l1_loss(
                pred_output.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        # 返回混合损失
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        return loss

    def get_l1_target(self,
                      l1_target,
                      gt,
                      stride,
                      x_shifts,
                      y_shifts,
                      eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        """评估单个样本的损失
        
        Args:
            batch_idx,              样本索引
            num_gt,                 真值目标个数
            total_num_anchors,      锚点数量
            gt_bboxes_per_image,    样本目标bbox
            gt_classes,             样本目标类别
            bboxes_preds_per_image, 预测bbox, [n_preds, 4]
            expanded_strides,
            x_shifts,               预测网格x坐标偏移量
            y_shifts,               预测网格y坐标偏移量
            cls_preds,              预测分类
            bbox_preds,             预测bbox
            obj_preds,              预测命中目标
            mode="gpu",  

        Returns:
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        
        """

        # if mode == "cpu":
        #     print("------------CPU Mode for This Batch-------------")
        #     gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
        #     bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
        #     gt_classes = gt_classes.cpu().float()
        #     expanded_strides = expanded_strides.cpu().float()
        #     x_shifts = x_shifts.cpu()
        #     y_shifts = y_shifts.cpu()

        # 将目标bbox嵌入成预测网格的mask
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # 计算IoU损失
        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image,
                                    bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(
            pair_wise_ious + 1e-8)  # [n_gt, n_valid_preds]

        # 计算分类损失
        gt_cls_per_image = (F.one_hot(gt_classes.to(
            torch.int64), cls_preds.shape[-1]).float().unsqueeze(1).repeat(
                1, num_in_boxes_anchor, 1))
        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(
                num_gt, 1, 1).sigmoid_() *
                          obj_preds_.float().unsqueeze(0).repeat(num_gt, 1,
                                                                 1).sigmoid_())
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        # 计算匹配代价
        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 *
                (~is_in_boxes_and_center))
        # 计算top k 匹配
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt,
                                    fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        # if mode == "cpu":
        #     # gt_matched_classes = gt_matched_classes.cuda()
        #     # fg_mask = fg_mask.cuda()
        #     # pred_ious_this_matching = pred_ious_this_matching.cuda()
        #     # matched_gt_inds = matched_gt_inds.cuda()
        #     gt_matched_classes = gt_matched_classes.cpu()
        #     fg_mask = fg_mask.cpu()
        #     pred_ious_this_matching = pred_ious_this_matching.cpu()
        #     matched_gt_inds = matched_gt_inds.cpu()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        """计算groundtruth嵌入预测网格的mask
        
        Args:
            gt_bboxes_per_image, 单样本真值bbox参数， [N, 4]
            expanded_strides,
            x_shifts,            预测网格x坐标偏移量
            y_shifts,            预测网格y坐标偏移量
            total_num_anchors,   锚点数量
            num_gt,              目标个数

        Returns:
            is_in_boxes_anchor,     网格命中目标mask
            is_in_boxes_and_center  网格命中目标中心mask

        """
        # 计算预测网格表示的中心点坐标
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image +
             0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_preds]
        y_centers_per_image = (
            (y_shifts_per_image +
             0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))

        # 计算目标bbox的坐标范围
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] -
             0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(
                 1, total_num_anchors))
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] +
             0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(
                 1, total_num_anchors))
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] -
             0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(
                 1, total_num_anchors))
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] +
             0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(
                 1, total_num_anchors))

        # 判断网格表示的中心点是否在目标bbox范围内
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # 计算目标中心的坐标范围
        # in fixed center
        center_radius = 2.5
        gt_bboxes_per_image_l = (
            gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
                1, total_num_anchors
            ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (
            gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
                1, total_num_anchors
            ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (
            gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
                1, total_num_anchors
            ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (
            gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
                1, total_num_anchors
            ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        # 判断网格点=表示的中心点是否命中目标bbox中心
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # 命中目标bbox， [n_preds,]
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        # 命中目标中心, [n_preds,]
        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor]
                                  & is_in_centers[:, is_in_boxes_anchor])
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt,
                           fg_mask):
        """动态K匹配
        
        Args:
            cost,             损失矩阵, [n_gt, n_preds]
            pair_wise_ious,   匹配IOU, [n_gt, n_prods]
            gt_classes,       真值目标的类别 [n_gt,]
            num_gt,           真值数量
            fg_mask           匹配mask, [n_preds,]

        Returns:
            num_fg,                     匹配成功的预测数, n_match_pred
            gt_matched_classes,         匹配成功的目标类别, [n_match_pred,]
            pred_ious_this_matching,    匹配成功的iou, [n_match_pred,]
            matched_gt_inds             匹配成功的目标索引, [n_match_pred, ]
        
        """
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx],
                                    k=dynamic_ks[gt_idx],
                                    largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix *
                                   pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        # for fo in fpn_outs:
        #     print('fpn_outs:{}', fo.detach().numpy().shape)
        outputs = self.head(fpn_outs)
        return outputs


from dataset.mscoco import COCODataset
from dataset.mscoco import COCOEvaluator
from dataset.data_augment import TrainTransform
from dataset.data_augment import ValTransform
from dataset.datasets_wrapper import YoloBatchSampler
from dataset.datasets_wrapper import DataLoader
from dataset.samplers import InfiniteSampler
from dataset.mosaic_detection import MosaicDetection
from learning_config.lr_scheduler import LRScheduler

import time


class Exp:

    def __init__(self,
                 data_dir='/Users/jielyu/Database/Dataset',
                 output_dir='./YOLOX_outputs',
                 batch_size=4,
                 num_data_worker=4) -> None:
        self.seed = None
        self.output_dir = output_dir

        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        self.act = 'silu'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = num_data_worker
        self.batch_size = batch_size
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = data_dir
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_model(self):
        """创建网络结构模型对象"""

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth,
                                 self.width,
                                 in_channels=in_channels,
                                 act=self.act)
            head = YOLOXHead(self.num_classes,
                             self.width,
                             in_channels=in_channels,
                             act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, no_aug=False, cache_img=False):
        """用于包装训练数据集"""
        # 创建MSCOCO数据集封装
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50,
                                   flip_prob=self.flip_prob,
                                   hsv_prob=self.hsv_prob),
            cache=cache_img,
        )
        # 添加样本增强
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120,
                                   flip_prob=self.flip_prob,
                                   hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        self.dataset = dataset
        # 创建采样器
        sampler = InfiniteSampler(len(self.dataset),
                                  seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        # 创建DataLoader
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler
        }
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(inputs,
                                               size=tsize,
                                               mode="bilinear",
                                               align_corners=False)
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_loss(self):
        """创建损失函数构建对象"""
        self.loss_net = YoloxLoss()
        return self.loss_net

    def get_optimizer(self, batch_size):
        """创建优化器对象"""
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(
                        v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(pg0,
                                        lr=lr,
                                        momentum=self.momentum,
                                        nesterov=True)
            optimizer.add_param_group({
                "params": pg1,
                "weight_decay": self.weight_decay
            })  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        """创建学习旅规划器"""
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, testdev=False, legacy=False):
        """创建验证集的数据载入器"""
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann
            if not testdev else "image_info_test-dev2017.json",
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size
        }
        val_loader = torch.utils.data.DataLoader(valdataset,
                                                 **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, testdev=False, legacy=False):
        """创建评估器"""
        val_loader = self.get_eval_loader(batch_size, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        """评估模型"""
        return evaluator.evaluate(model, is_distributed, half)

    @staticmethod
    def get_latest_file(file_dir, cont='yolox'):
        """用于获取指定目录下的最新文件
        
        Args:
            file_dir: 指定目录
            cont: 文件名包含的内容

        Returns:
            file_path: 最新文件的路径

        """
        if not os.path.isdir(file_dir):
            return None
        file_list = os.listdir(file_dir)
        file_list.sort(
            key=lambda fn: os.path.getmtime(os.path.join(file_dir, fn))
            if not os.path.isdir(os.path.join(file_dir, fn)) else 0)
        file_path = None
        for filename in file_list:
            if cont in filename:
                file_path = os.path.join(file_dir, filename)
        return file_path

    @classmethod
    def resume_model(cls, load_dir, model, optimizer, device='cuda'):
        """恢复训练状态"""
        # 查找最新模型文件
        start_epoch = -1
        lastest_model_path = cls.get_latest_file(load_dir, 'yolox')
        if lastest_model_path is not None and os.path.isfile(
                lastest_model_path):
            # 载入模型文件
            ckpt = torch.load(lastest_model_path,
                              map_location=torch.device(device))
            if 'model' in ckpt and 'optimizer' in ckpt and 'start_epoch' in ckpt:
                # 恢复状态
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt['start_epoch']
                logger.info('resume model at epoch {} from {} succ'.format(
                    start_epoch, lastest_model_path))
        return start_epoch

    @staticmethod
    def save_model(save_dir, model, optimizer, epoch, loss):
        """保存模型训练状态到文件"""
        # 构造需要存储的模型对象
        ckpt_state = {
            "start_epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        # 检查并生成目录
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # 保存到文件
        filename = os.path.join(save_dir,
                                'yolox_epoch_{}_ckpt.pth'.format(epoch))
        torch.save(ckpt_state, filename)
        return filename

    def train(self, is_gpu=False, no_aug=False):
        """用于训练模型"""
        logger.info('starting train model in {} mode ...'.format(
            'CUDA' if is_gpu else 'CPU'))
        # 创建模型
        model = self.get_model()
        loss_net = self.get_loss()
        model.train()
        if is_gpu is True:
            # 该语句必须在创建optimizer之前
            model = model.cuda()
        # 创建优化器
        optimizer = self.get_optimizer(self.batch_size)
        # 创建数据载入器
        train_loader = self.get_data_loader(self.batch_size, no_aug)
        eval_loader = self.get_eval_loader(self.batch_size)
        num_train_iters = len(train_loader)
        num_eval_iters = len(eval_loader)
        # 创建学习率规划对象
        lr_scheduler = self.get_lr_scheduler(
            self.basic_lr_per_img * self.batch_size, num_train_iters)
        # 断点接续训练
        device = 'cuda' if is_gpu is True else 'cpu'
        start_epoch = 1 + self.resume_model(self.output_dir, model, optimizer,
                                            device)

        # 训练迭代
        for idx_epoch in range(start_epoch, self.max_epoch):
            for idx_iter, batch in enumerate(train_loader):
                idx_iter = idx_iter % num_train_iters
                # 数据转换
                images, labels = batch[0], batch[1]
                if is_gpu is True:
                    images = images.cuda()
                    labels = labels.cuda()
                # 前向推理
                forward_start = time.time()
                outputs = model(images)
                loss = loss_net(*outputs, labels=labels)
                forword_cost = time.time() - forward_start
                # 反向传播
                backward_start = time.time()
                optimizer.zero_grad()  # 缓存梯度清零
                loss.backward()  # 误差反传
                optimizer.step()  # 更新权重
                backward_cost = time.time() - backward_start
                # 更新学习率
                lr = lr_scheduler.update_lr(idx_iter +
                                            idx_epoch * num_train_iters + 1)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                loss_value = loss.cpu().detach().numpy()
                # 打印日志
                if idx_iter % 100 == 0:
                    logger.info(
                        '[{}/{}][{}/{}] train phase. loss={:.4f}, timecost=[f:{:.4f}, b:{:.4f}]'
                        .format(idx_epoch, self.max_epoch, idx_iter,
                                num_train_iters, loss_value, forword_cost,
                                backward_cost))
                if (idx_iter + 1 == num_train_iters):
                    break
            # 保存模型
            model_path = self.save_model(self.output_dir, model, optimizer,
                                         idx_epoch, loss_value)
            logger.info('save {}-epoch model to {}'.format(
                idx_epoch, model_path))

    @staticmethod
    def postprocess(prediction,
                    num_classes,
                    conf_thre=0.7,
                    nms_thre=0.45,
                    class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:,
                                                          5:5 + num_classes],
                                               1,
                                               keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >=
                         conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def demo(self,
             img_file,
             model_file,
             conf_thre=0.25,
             nms_thre=0.45,
             is_gpu=False):
        """用于测试检测效果的demo"""
        logger.info('starting load model in {} mode ...'.format(
            'CUDA' if is_gpu else 'CPU'))
        device = 'cuda' if is_gpu is True else 'cpu'

        # 创建模型
        model = self.get_model()
        model.eval()
        if is_gpu is True:
            # 该语句必须在创建optimizer之前
            model = model.cuda()
        # 载入模型文件
        if not os.path.isfile(model_file):
            raise ValueError('not found model from:' + model_file)
        ckpt = torch.load(model_file, map_location=torch.device(device))
        if 'model' not in ckpt:
            raise ValueError('model filed not exist')
        model.load_state_dict(ckpt['model'])
        # 载入图片
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError('failed to load image from:' + img_file)
        img_resized = cv2.resize(img, (640, 640))
        print(img_resized.shape)
        # 数据格式准备
        img_input = torch.from_numpy(img_resized)
        img_input = img_input.unsqueeze(dim=0)
        img_input = img_input.permute(0, 3, 1, 2).float()
        print(img_input.shape, img_input.dtype)
        if is_gpu is True:
            img_input = img_input.cuda()
        # 推理过程
        outputs = model(img_input)
        preds = outputs[0]
        # NMS
        preds = Exp.postprocess(preds,
                                80,
                                conf_thre=conf_thre,
                                nms_thre=nms_thre)
        # 显示
        for idx, detections in enumerate(preds):
            plt.clf()
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_show)
            gca = plt.gca()
            for i in range(detections.shape[0]):
                det = detections[i].detach().numpy()
                scales = [
                    img.shape[1] / img_resized.shape[1],
                    img.shape[0] / img_resized.shape[0]
                ]
                det[0] = det[0] * scales[0]
                det[1] = det[1] * scales[1]
                det[2] = det[2] * scales[0]
                det[3] = det[3] * scales[1]
                rect = plt.Rectangle((det[0], det[1]),
                                     det[2] - det[0],
                                     det[3] - det[1],
                                     fill=False,
                                     edgecolor='red',
                                     linewidth=1)
                gca.add_patch(rect)
                plt.text(det[0],
                         det[1],
                         COCODataset.COCO_CLASSES[round(det[6])],
                         color='blue')
            plt.show()


def str2bool(v):
    """用于命令行解析bool类型的参数"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# import matplotlib.pyplot as plt
import argparse


def parse_args():
    args = argparse.ArgumentParser(description='yolox')
    args.add_argument('--phase',
                      type=str,
                      default='train',
                      choices=['train', 'demo'])
    args.add_argument('--img-path', type=str, default=None)
    args.add_argument('--gpu', type=str2bool, default=False)
    args.add_argument('--colab', type=str2bool, default=False)
    args.add_argument('--data_dir',
                      type=str,
                      default='/Users/jielyu/Database/Dataset')
    args.add_argument('--output_dir', type=str, default='./YOLOX_outputs')
    args.add_argument('--batch_size', type=int, default=4)
    args.add_argument('--num_data_worker', type=int, default=4)
    args.add_argument('--no_aug', type=str2bool, default=False)
    args.add_argument('--conf-thresh', type=float, default=0.25)
    args.add_argument('--nms-thresh', type=float, default=0.45)
    return args.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    # 设置参数
    if args.colab is True:
        args.data_dir = './dataset'
        args.output_dir = '/content/drive/Shareddrives/jielyu-5T/output/yolox_exp'
    # 创建实验对象
    exp = Exp(data_dir=args.data_dir,
              output_dir=args.output_dir,
              batch_size=args.batch_size,
              num_data_worker=args.num_data_worker)
    # 执行训练流程
    if args.phase == 'train':
        exp.train(args.gpu, args.no_aug)
    elif args.phase == 'demo':
        img_file = './data/mscoco-sample/000000000139.jpg'
        model_file = './YOLOX_outputs/yolox_l.pth'
        exp.demo(img_file,
                 model_file,
                 conf_thre=args.conf_thresh,
                 nms_thre=args.nms_thresh)


if __name__ == '__main__':
    main()
