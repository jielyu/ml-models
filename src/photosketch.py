# encoding: utf-8

"""
参考代码：https://github.com/ml4a/ml4a
参考代码：https://github.com/mtli/PhotoSketch
"""

import os
import random
import functools
import time
from types import SimpleNamespace
import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.mobile_optimizer import optimize_for_mobile
from PIL import Image


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    print("initialization method [%s]" % init_type)
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [%s] is not implemented" % init_type
        )


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.lr_policy
        )
    return scheduler


def define_G(
    input_nc,
    output_nc,
    ngf,
    which_model_netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == "resnet_9blocks":
        netG = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
        )
    elif which_model_netG == "resnet_6blocks":
        netG = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=6,
        )
    elif which_model_netG == "unet_128":
        netG = UnetGenerator(
            input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    elif which_model_netG == "unet_256":
        netG = UnetGenerator(
            input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    else:
        raise NotImplementedError(
            "Generator model name [%s] is not recognized" % which_model_netG
        )

    init_weights(netG, init_type=init_type)
    return netG


def define_D(
    input_nc,
    ndf,
    which_model_netD,
    n_layers_D=3,
    norm="batch",
    use_sigmoid=False,
    init_type="normal",
):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == "basic":
        netD = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif which_model_netD == "n_layers":
        netD = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif which_model_netD == "pixel":
        netD = PixelDiscriminator(
            input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif which_model_netD == "global":
        netD = GlobalDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif which_model_netD == "global_np":
        netD = GlobalNPDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    else:
        raise NotImplementedError(
            "Discriminator model name [%s] is not recognized" % which_model_netD
        )
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, device="cpu"
    ):
        super(GANLoss, self).__init__()
        self.device = device
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss().to(device)
        else:
            self.loss = nn.BCELoss().to(device)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (
                self.real_label_var.numel() != input.numel()
            )
            if create_label:
                self.real_label_var = torch.full(
                    input.size(),
                    self.real_label,
                    requires_grad=False,
                    device=self.device,
                )
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (
                self.fake_label_var.numel() != input.numel()
            )
            if create_label:
                self.fake_label_var = torch.full(
                    input.size(),
                    self.fake_label,
                    requires_grad=False,
                    device=self.device,
                )
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False
    ):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 256
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        # 128

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        # 64
        # 32

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        # 31

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        # 30

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class GlobalDiscriminator(nn.Module):
    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False
    ):
        super(GlobalDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 256
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        # 128
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        # 64
        # 32

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=2,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        # 16

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=0)]
        sequence += [nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=0)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class GlobalNPDiscriminator(nn.Module):
    # no padding
    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False
    ):
        super(GlobalNPDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 256
        kw = [8, 3, 4]
        padw = 0
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw[0], stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        # 125
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw[n],
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        # 62
        # 30

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=2,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        # 14

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=2, padding=0)]
        # 6
        sequence += [nn.Conv2d(1, 1, kernel_size=6, stride=1, padding=0, bias=use_bias)]
        # 1

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class BaseModel:
    def name(self):
        return "BaseModel"

    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device("cuda" if opt.use_cuda else "cpu")
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        network = network.to(self.device)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        if self.opt.pretrain_path:
            save_path = os.path.join(self.opt.pretrain_path, save_filename)
        else:
            save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class Pix2PixModel(BaseModel):
    def name(self):
        return "Pix2PixModel"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.which_model_netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D,
                opt.norm,
                use_sigmoid,
                opt.init_type,
            )

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, "G", opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, "D", opt.which_epoch)

        self.netG = self.netG.to(self.device)
        if self.isTrain:
            self.netD = self.netD.to(self.device)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, device=self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, opt))

        print("---------- Networks initialized -------------")
        print_network(self.netG)
        if self.isTrain:
            print_network(self.netD)
        print("-----------------------------------------------")

    def set_input(self, input):
        AtoB = self.opt.which_direction == "AtoB"
        self.input_A = input["A" if AtoB else "B"].to(self.device)
        self.input_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        if "w" in input:
            self.input_w = input["w"]
        if "h" in input:
            self.input_h = input["h"]

    def forward(self):
        self.real_A = self.input_A
        print(
            "input: ",
            torch.mean(self.real_A),
            torch.std(self.real_A),
            self.real_A.dtype,
        )
        self.fake_B = self.netG(self.real_A)
        print("after", torch.mean(self.fake_B), torch.std(self.fake_B))
        self.real_B = self.input_B

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(
            torch.cat((self.real_A, self.fake_B), 1).detach()
        )
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        n = self.real_B.shape[1]
        loss_D_real_set = torch.empty(n, device=self.device)
        for i in range(n):
            sel_B = self.real_B[:, i, :, :].unsqueeze(1)
            real_AB = torch.cat((self.real_A, sel_B), 1)
            pred_real = self.netD(real_AB)
            loss_D_real_set[i] = self.criterionGAN(pred_real, True)
        self.loss_D_real = torch.mean(loss_D_real_set)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_G

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        # Second, G(A) = B
        n = self.real_B.shape[1]
        fake_B_expand = self.fake_B.expand(-1, n, -1, -1)
        L1 = torch.abs(fake_B_expand - self.real_B)
        L1 = L1.view(-1, n, self.real_B.shape[2] * self.real_B.shape[3])
        L1 = torch.mean(L1, 2)
        min_L1, min_idx = torch.min(L1, 1)
        self.loss_G_L1 = torch.mean(min_L1) * self.opt.lambda_A
        self.min_idx = min_idx

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict(
            [
                ("G_GAN", self.loss_G_GAN.item()),
                ("G_L1", self.loss_G_L1.item()),
                ("D_real", self.loss_D_real.item()),
                ("D_fake", self.loss_D_fake.item()),
            ]
        )

    def get_current_visuals(self):
        real_A = tensor2im(self.real_A.detach())
        fake_B = tensor2im(self.fake_B.detach())
        if self.isTrain:
            sel_B = self.real_B[:, self.min_idx[0], :, :]
        else:
            sel_B = self.real_B[:, 0, :, :]
        real_B = tensor2im(sel_B.unsqueeze(1).detach())
        return OrderedDict([("real_A", real_A), ("fake_B", fake_B), ("real_B", real_B)])

    def save(self, label):
        self.save_network(self.netG, "G", label)
        self.save_network(self.netD, "D", label)

    def write_image(self, out_dir):
        image_numpy = self.fake_B.detach()[0][0].cpu().float().numpy()
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
        image_pil = image_pil.resize((self.input_w[0], self.input_h[0]), Image.BICUBIC)
        name, _ = os.path.splitext(os.path.basename(self.image_paths[0]))
        out_path = os.path.join(out_dir, name + self.opt.suffix + ".png")
        image_pil.save(out_path)


class PhotoSketch:
    def __init__(self, model_dir="models/ml4a_photosketch/pretrained", use_gpu=False):
        opt = {}
        opt = SimpleNamespace(**opt)
        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.name = model_dir
        opt.checkpoints_dir = "."
        opt.model = "pix2pix"
        opt.which_direction = "AtoB"
        opt.norm = "batch"
        opt.input_nc = 3
        opt.output_nc = 1
        opt.which_model_netG = "resnet_9blocks"
        opt.no_dropout = True
        opt.isTrain = False
        opt.use_cuda = use_gpu
        opt.ngf = 64
        opt.ndf = 64
        opt.init_type = "normal"
        opt.which_epoch = "latest"
        opt.pretrain_path = model_dir
        self.model = self.create_model(opt)

    @staticmethod
    def create_model(opt):
        model = Pix2PixModel()
        model.initialize(opt)
        print("model [%s] was created" % (model.name()))
        return model

    def __call__(self, img):

        img = np.array(img) / 255.0
        h, w = img.shape[0:2]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()  # .to(device)
        data = {"A_paths": "", "A": img, "B": img}
        self.model.set_input(data)
        self.model.test()
        output = tensor2im(self.model.fake_B)
        return output


def main():
    img_path = "data/photosketch-samples/Valley-Taurus-Mountains-Turkey.jpg"
    ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    photosketch = PhotoSketch()
    start_time = time.time()
    img_sketch = photosketch(img)
    end_time = time.time()
    print("Cost Time: %.2f sec" % (end_time - start_time))

    # 可视化
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(img_sketch)
    plt.axis("off")
    plt.title("Sketch Image")
    plt.savefig("output/photosketch.png", bbox_inches="tight", dpi=300)
    plt.show()


def pth2pt():
    # 载入模型
    model_path = "models/ml4a_photosketch/pretrained/latest_net_G.pth"
    device = torch.device("cpu")
    model = define_G(3, 1, 64, "resnet_9blocks", "batch", False, "normal")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    # 使用eval后结果异常
    # model.eval()
    # print_network(model)
    # 读取图片
    img_path = "data/photosketch-samples/Valley-Taurus-Mountains-Turkey.jpg"
    ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    normal_img = np.array(rgb) / 255.0
    img = np.transpose(normal_img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    with torch.no_grad():
        # 测试通路是否正常
        print("input: ", torch.mean(img), torch.std(img), img.dtype)
        output = model(img)
        print("after", torch.mean(output), torch.std(output))
        # 导出libtorch所需的pt文件
        input = torch.rand(1, 3, 1067, 1600)
        traced_script_module = torch.jit.trace(model, input)
        traced_script_module.save("output/photo_sketch.pt")
        # 导出lite所需的ptl文件
        optimize_for_mobile(traced_script_module)._save_for_lite_interpreter(
            "output/photo_sketch.ptl"
        )
    # 可视化
    img_sketch = tensor2im(output)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(img_sketch)
    plt.axis("off")
    plt.title("Sketch Image")
    plt.show()


if __name__ == "__main__":
    main()
    # pth2pt()
