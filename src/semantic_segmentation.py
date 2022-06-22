# encoding: utf-8

"""
参考代码：https://github.com/ml4a/ml4a
参考代码：https://github.com/CSAILVision/semantic-segmentation-pytorch
"""

import csv
import math
import os
import time
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
from scipy.io import loadmat
from common_utils.download_utils import download_data_file, download_text_file

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


from semantic_segmentation_utils import resnet, resnext, mobilenet, hrnet
from semantic_segmentation_utils.batchnorm import SynchronizedBatchNorm2d

BatchNorm2d = SynchronizedBatchNorm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None:  # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(
                    self.encoder(feed_dict["img_data"], return_feature_maps=True)
                )
            else:
                pred = self.decoder(
                    self.encoder(feed_dict["img_data"], return_feature_maps=True)
                )

            loss = self.crit(pred, feed_dict["seg_label"])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict["seg_label"])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict["seg_label"])
            return loss, acc
        # inference
        else:
            pred = self.decoder(
                self.encoder(feed_dict["img_data"], return_feature_maps=True),
                segSize=segSize,
            )
            return pred


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)
        # elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch="resnet50dilated", fc_dim=512, weights=""):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == "mobilenetv2dilated":
            orig_mobilenet = mobilenet.__dict__["mobilenetv2"](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == "resnet18":
            orig_resnet = resnet.__dict__["resnet18"](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == "resnet18dilated":
            orig_resnet = resnet.__dict__["resnet18"](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == "resnet34":
            raise NotImplementedError
            orig_resnet = resnet.__dict__["resnet34"](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == "resnet34dilated":
            raise NotImplementedError
            orig_resnet = resnet.__dict__["resnet34"](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == "resnet50":
            orig_resnet = resnet.__dict__["resnet50"](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == "resnet50dilated":
            orig_resnet = resnet.__dict__["resnet50"](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == "resnet101":
            orig_resnet = resnet.__dict__["resnet101"](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == "resnet101dilated":
            orig_resnet = resnet.__dict__["resnet101"](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == "resnext101":
            orig_resnext = resnext.__dict__["resnext101"](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext)  # we can still use class Resnet
        elif arch == "hrnetv2":
            net_encoder = hrnet.__dict__["hrnetv2"](pretrained=pretrained)
        else:
            raise Exception("Architecture undefined!")

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print("Loading weights for net_encoder")
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage),
                strict=False,
            )
        return net_encoder

    @staticmethod
    def build_decoder(
        arch="ppm_deepsup", fc_dim=512, num_class=150, weights="", use_softmax=False
    ):
        arch = arch.lower()
        if arch == "c1_deepsup":
            net_decoder = C1DeepSup(
                num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax
            )
        elif arch == "c1":
            net_decoder = C1(
                num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax
            )
        elif arch == "ppm":
            net_decoder = PPM(
                num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax
            )
        elif arch == "ppm_deepsup":
            net_decoder = PPMDeepsup(
                num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax
            )
        elif arch == "upernet_lite":
            net_decoder = UPerNet(
                num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax, fpn_dim=256
            )
        elif arch == "upernet":
            net_decoder = UPerNet(
                num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax, fpn_dim=512
            )
        else:
            raise Exception("Architecture undefined!")

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print("Loading weights for net_decoder")
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage),
                strict=False,
            )
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(
        self, num_class=150, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6)
    ):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                fc_dim + len(pool_scales) * 512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1),
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(conv5),
                    (input_size[2], input_size[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(
        self, num_class=150, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6)
    ):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                fc_dim + len(pool_scales) * 512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1),
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(conv5),
                    (input_size[2], input_size[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(
        self,
        num_class=150,
        fc_dim=4096,
        use_softmax=False,
        pool_scales=(1, 2, 3, 6),
        fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256,
    ):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1
        )

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(
                nn.Sequential(
                    conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
                )
            )
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1),
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind="mergesort" if return_index else "quicksort")
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode="RGB"):
    labelmap = labelmap.astype("int")
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * np.tile(
            colors[label], (labelmap.shape[0], labelmap.shape[1], 1)
        )

    if mode == "BGR":
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


class SemanticSegmentation:
    @staticmethod
    def download_models():
        model_dir = "models/ml4a_semantic_segmentation"
        color_path = download_data_file(
            "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/color150.mat",
            os.path.join(model_dir, "color150.mat"),
        )
        data_path = download_text_file(
            "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/object150_info.csv",
            os.path.join(model_dir, "object150_info.csv"),
        )
        encoder_model = download_data_file(
            "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth",
            os.path.join(model_dir, "encoder_epoch_20.pth"),
        )
        decoder_model = download_data_file(
            "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth",
            os.path.join(model_dir, "decoder_epoch_20.pth"),
        )

    def __init__(
        self, model_dir="models/ml4a_semantic_segmentation", use_gpu=False
    ) -> None:
        self.use_gpu = use_gpu
        self.color_path = os.path.join(model_dir, "color150.mat")
        self.data_path = os.path.join(model_dir, "object150_info.csv")
        # colors and class names
        self.colors = loadmat(self.color_path)["colors"]
        self.classes = {}
        with open(self.data_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.classes[int(row[0]) - 1] = row[5].split(";")[0]

        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch="resnet50dilated",
            fc_dim=2048,
            weights=os.path.join(model_dir, "encoder_epoch_20.pth"),
        )

        net_decoder = ModelBuilder.build_decoder(
            arch="ppm_deepsup",
            fc_dim=2048,
            num_class=150,
            weights=os.path.join(model_dir, "decoder_epoch_20.pth"),
            use_softmax=True,
        )

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.model = SegmentationModule(net_encoder, net_decoder, crit)
        self.model.eval()
        if use_gpu:
            self.model.cuda()

    def get_class_index(self, class_name):
        if class_name not in self.classes.values():
            return None
        return list(self.classes.keys())[list(self.classes.values()).index(class_name)]

    def get_color_labels(self, pred):
        im_vis = colorEncode(pred, self.colors).astype(np.uint8)
        return im_vis

    def encode_color_mask(self, pred):
        return colorEncode(pred, self.colors).astype(np.uint8)

    def visualize(self, img, pred, index=None, concat_original=True):
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
        im_vis = colorEncode(pred, self.colors).astype(np.uint8)
        if concat_original:
            im_vis = np.concatenate((img, im_vis), axis=1)
        return np.array(Image.fromarray(im_vis))

    def get_mask(self, pred, index):
        is_list = isinstance(index, list)
        index = index if is_list else [index]
        index = [
            self.get_class_index(idx) if isinstance(idx, str) else idx for idx in index
        ]
        h, w = pred.shape[:2]
        mask = np.zeros((h, w, len(index)))
        for i, idx in enumerate(index):
            mask_channel = pred.copy()
            mask_channel[pred != idx] = 0
            mask_channel[pred == idx] = 255
            mask[:, :, i] = mask_channel
        if len(index) == 1:
            mask = mask[:, :, 0]
        return mask.astype(np.uint8)

    def __call__(self, imgs):

        pil_to_tensor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
                    std=[0.229, 0.224, 0.225],
                ),  # across a large photo dataset.
            ]
        )
        img_data = pil_to_tensor(imgs)
        singleton_batch = {
            "img_data": img_data[None].cuda() if self.use_gpu else img_data[None]
        }

        # Run the segmentation at the highest resolution
        with torch.no_grad():
            scores = self.model(singleton_batch, segSize=img_data.shape[1:])

        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()
        return pred


def main():
    img_path = "data/semantic_segmentation-samples/3.jpg"
    ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    semantic_segmentation = SemanticSegmentation()
    start_time = time.time()
    pred = semantic_segmentation(img)
    print("Cost Time(s): {:.2f}".format(time.time() - start_time))
    color_mask = semantic_segmentation.encode_color_mask(pred)

    # 可视化
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.title("Semantic Segmentation")
    plt.axis("off")
    plt.savefig("output/semantic_segmentation.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    # SemanticSegmentation.download_models()
