# encoding: utf-8

import importlib
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from PIL import Image
from common_utils.download_utils import download_from_gdrive

# from models.networks.architecture import VGG19

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self,
        gan_mode,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
        opt=None,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == "ls":
            pass
        elif gan_mode == "original":
            pass
        elif gan_mode == "w":
            pass
        elif gan_mode == "hinge":
            pass
        else:
            raise ValueError("Unexpected gan_mode {}".format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == "original":  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == "ls":
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            "Network [%s] was created. Total number of parameters: %.1f million. "
            "To see the architecture, do print(network)."
            % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if ss == target_width:
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class NetworkUtils:
    @staticmethod
    def find_class_in_module(target_cls_name, module):
        target_cls_name = target_cls_name.replace("_", "").lower()
        clslib = importlib.import_module(module)
        cls = None
        for name, clsobj in clslib.__dict__.items():
            if name.lower() == target_cls_name:
                cls = clsobj

        if cls is None:
            print(
                "In %s, there should be a class whose name matches %s in lowercase without underscore(_)"
                % (module, target_cls_name)
            )
            exit(0)

        return cls

    @staticmethod
    def find_network_using_name(target_network_name, filename):
        target_class_name = target_network_name + filename
        module_name = "models.networks." + filename
        network = NetworkUtils.find_class_in_module(target_class_name, module_name)

        assert issubclass(network, BaseNetwork), (
            "Class %s should be a subclass of BaseNetwork" % network
        )

        return network

    @staticmethod
    def modify_commandline_options(parser, is_train):
        opt, _ = parser.parse_known_args()

        netG_cls = NetworkUtils.find_network_using_name(opt.netG, "generator")
        parser = netG_cls.modify_commandline_options(parser, is_train)
        if is_train:
            netD_cls = NetworkUtils.find_network_using_name(opt.netD, "discriminator")
            parser = netD_cls.modify_commandline_options(parser, is_train)
        netE_cls = NetworkUtils.find_network_using_name("conv", "encoder")
        parser = netE_cls.modify_commandline_options(parser, is_train)

        return parser

    @staticmethod
    def create_network(cls, opt):
        net = cls(opt)
        net.print_network()
        if len(opt.gpu_ids) > 0:
            assert torch.cuda.is_available()
            net.cuda()
        net.init_weights(opt.init_type, opt.init_variance)
        return net

    @staticmethod
    def define_G(opt):
        netG_cls = NetworkUtils.find_network_using_name(opt.netG, "generator")
        return NetworkUtils.create_network(netG_cls, opt)

    @staticmethod
    def define_D(opt):
        netD_cls = NetworkUtils.find_network_using_name(opt.netD, "discriminator")
        return NetworkUtils.create_network(netD_cls, opt)

    @staticmethod
    def define_E(opt):
        # there exists only one encoder type
        netE_cls = NetworkUtils.find_network_using_name("conv", "encoder")
        return NetworkUtils.create_network(netE_cls, opt)

    @staticmethod
    def save_network(net, label, epoch, opt):
        save_filename = "%s_net_%s.pth" % (epoch, label)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        if len(opt.gpu_ids) and torch.cuda.is_available():
            net.cuda()

    @staticmethod
    def load_network(net, label, epoch, opt):
        save_filename = "%s_net_%s.pth" % (epoch, label)
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        save_path = os.path.join(save_dir, save_filename)
        weights = torch.load(save_path)
        net.load_state_dict(weights)
        return net

    @staticmethod
    def get_params(opt, size):
        w, h = size
        new_h = h
        new_w = w
        if opt.preprocess_mode == "resize_and_crop":
            new_h = new_w = opt.load_size
        elif opt.preprocess_mode == "scale_width_and_crop":
            new_w = opt.load_size
            new_h = opt.load_size * h // w
        elif opt.preprocess_mode == "scale_shortside_and_crop":
            ss, ls = min(w, h), max(w, h)  # shortside and longside
            width_is_shorter = w == ss
            ls = int(opt.load_size * ls / ss)
            new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

        flip = random.random() > 0.5
        return {"crop_pos": (x, y), "flip": flip}

    @staticmethod
    def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
        transform_list = []
        if "resize" in opt.preprocess_mode:
            osize = [opt.load_size, opt.load_size]
            transform_list.append(transforms.Resize(osize, interpolation=method))
        elif "scale_width" in opt.preprocess_mode:
            transform_list.append(
                transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method))
            )
        elif "scale_shortside" in opt.preprocess_mode:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __scale_shortside(img, opt.load_size, method)
                )
            )

        if "crop" in opt.preprocess_mode:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __crop(img, params["crop_pos"], opt.crop_size)
                )
            )

        if opt.preprocess_mode == "none":
            base = 32
            transform_list.append(
                transforms.Lambda(lambda img: __make_power_2(img, base, method))
            )

        if opt.preprocess_mode == "fixed":
            w = opt.crop_size
            h = round(opt.crop_size / opt.aspect_ratio)
            transform_list.append(
                transforms.Lambda(lambda img: __resize(img, w, h, method))
            )

        if opt.isTrain and not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params["flip"]))
            )

        if toTensor:
            transform_list += [transforms.ToTensor()]

        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):

        NetworkUtils.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = (
            torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        )
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt
            )
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == "generator":
            g_loss, generated = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated
        elif mode == "discriminator":
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == "encode_only":
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == "inference":
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        NetworkUtils.save_network(self.netG, "G", epoch, self.opt)
        NetworkUtils.save_network(self.netD, "D", epoch, self.opt)
        if self.opt.use_vae:
            NetworkUtils.save_network(self.netE, "E", epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = NetworkUtils.define_G(opt)
        netD = NetworkUtils.define_D(opt) if opt.isTrain else None
        netE = NetworkUtils.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = NetworkUtils.load_network(netG, "G", opt.which_epoch, opt)
            if opt.isTrain:
                netD = NetworkUtils.load_network(netD, "D", opt.which_epoch, opt)
            if opt.use_vae:
                netE = NetworkUtils.load_network(netE, "E", opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data["label"] = data["label"].long()
        if self.use_gpu():
            data["label"] = data["label"].cuda()
            data["instance"] = data["instance"].cuda()
            data["image"] = data["image"].cuda()

        # create one-hot label map
        label_map = data["label"]
        bs, _, h, w = label_map.size()
        nc = (
            self.opt.label_nc + 1
            if self.opt.contain_dontcare_label
            else self.opt.label_nc
        )
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data["instance"]
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data["image"]

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae
        )

        if self.opt.use_vae:
            G_losses["KLD"] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image
        )

        G_losses["GAN"] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()
                    )
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses["GAN_Feat"] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses["VGG"] = (
                self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
            )

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image
        )

        D_losses["D_Fake"] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses["D_real"] = self.criterionGAN(pred_real, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (
            not compute_kld_loss
        ) or self.opt.use_vae, "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2 :]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Spade:
    def __init__(self, model_dir="models/ml4a_spade", model_name="landscope") -> None:
        checkpoint_dir = os.path.join(model_dir, model_name)
        with open(
            os.path.join(checkpoint_dir, "classes_list.txt"), "r"
        ) as classes_file:
            self.classes = eval(classes_file.read())
        opt_file = os.path.join(checkpoint_dir, "opt.txt")
        parsed_opt = self.parse_opt_file(opt_file)
        opt = EasyDict({})
        opt.isTrain = False
        opt.checkpoints_dir = model_dir
        opt.name = model_name
        opt.aspect_ratio = float(parsed_opt["aspect_ratio"])
        opt.load_size = int(parsed_opt["load_size"])
        opt.crop_size = int(parsed_opt["crop_size"])
        opt.no_instance = True if parsed_opt["no_instance"] == "True" else False
        opt.preprocess_mode = parsed_opt["preprocess_mode"]
        opt.contain_dontcare_label = (
            True if parsed_opt["contain_dontcare_label"] == "True" else False
        )
        opt.gpu_ids = parsed_opt["gpu_ids"]
        opt.netG = parsed_opt["netG"]
        opt.ngf = int(parsed_opt["ngf"])
        opt.num_upsampling_layers = parsed_opt["num_upsampling_layers"]
        opt.use_vae = True if parsed_opt["use_vae"] == "True" else False
        opt.label_nc = int(parsed_opt["label_nc"])
        opt.semantic_nc = (
            opt.label_nc
            + (1 if opt.contain_dontcare_label else 0)
            + (0 if opt.no_instance else 1)
        )
        opt.norm_G = parsed_opt["norm_G"]
        opt.init_type = parsed_opt["init_type"]
        opt.init_variance = float(parsed_opt["init_variance"])
        opt.which_epoch = parsed_opt["which_epoch"]
        self.opt = opt
        self.model = Pix2PixModel(opt)
        self.model.eval()

    @staticmethod
    def parse_opt_file(path):
        file = open(path, "rb")
        opt = {}
        for line in file.readlines():
            line = str(line).split(": ")
            key = line[0].split(" ")[-1]
            value = line[1].split(" ")[0]
            opt[key] = value
        return opt

    @staticmethod
    def get_class_index(classes, class_name):
        if class_name not in classes.values():
            return None
        return list(classes.keys())[list(classes.values()).index(class_name)]

    @staticmethod
    def download_models():
        pretrained_models = {
            "cityscapes": [
                ["1_APZIT-3eD8KXK6GFz4cFXpN2qlQbez1", "latest_net_G.pth"],
                ["1zIxWGADWABWQRdXZqTWlzg7RwHfWZLt9", "opt.txt"],
                ["1Wn-OAagSYZplZJusvK9In8WtqwjS_ij2", "classes_list.txt"],
            ],
            "ade20k": [
                ["1shvEumc5PrqXIahV61_fLRNOUTb96wSg", "latest_net_G.pth"],
                ["1JbjQj7AdHgFrCRQFPTADybLS7AzyoyJ1", "opt.txt"],
                ["1FQ59iTkQ3fSnjEuaYsn7WdBB_j1fAOhE", "classes_list.txt"],
            ],
            "coco": [
                ["16KfJKje4aNUQSAxmzzKowJxpPzYgUOlo", "latest_net_G.pth"],
                ["1Ed16m6SAZNoQwSA2-fGYOhqWWU671M47", "opt.txt"],
                ["1XukXJvb2tYbEcvSCenRDFHkypdUJUj1Q", "classes_list.txt"],
            ],
            "landscapes": [
                ["15VSa2m2F6Ch0NpewDR7mkKAcXlMgDi5F", "latest_net_G.pth"],
                ["1zm26Oct3KaqO0dTW2dC_lNg8awsyo9sP", "opt.txt"],
                ["1jsgr-6TZHDFll9ZdszpY8JNyY6B_5MzI", "classes_list.txt"],
            ],
        }
        for model_name, models in pretrained_models.items():
            model_subfolder = os.path.join("models/ml4a_spade", model_name)
            for gdrive_id, filename in pretrained_models[model_name]:
                location = os.path.join(model_subfolder, filename)
                download_from_gdrive(gdrive_id, location)

    def __call__(self, labelmap):
        labelmap = Image.fromarray(np.array(labelmap).astype(np.uint8))
        params = NetworkUtils.get_params(self.opt, labelmap.size)
        transform_label = NetworkUtils.get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False
        )
        label_tensor = transform_label(labelmap) * 255.0
        label_tensor[label_tensor == 255.0] = self.opt.label_nc
        transform_image = NetworkUtils.get_transform(self.opt, params)
        image_tensor = transform_image(Image.new("RGB", (500, 500)))
        data = {
            "label": label_tensor.unsqueeze(0),
            "instance": label_tensor.unsqueeze(0),
            "image": image_tensor.unsqueeze(0),
        }
        generated = self.model(data, mode="inference")
        output = NetworkUtils.tensor2im(generated[0])
        output = Image.fromarray(output)
        return output


def main():
    spade = Spade()


if __name__ == "__main__":
    # main()
    Spade.download_models()
