# encoding: utf-8

"""
参考代码：https://github.com/ml4a/ml4a
参考代码：https://github.com/genekogan/SPADE
"""

import importlib
import os
import random
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from PIL import Image
from common_utils.download_utils import download_from_gdrive
from semantic_segmentation import SemanticSegmentation

import spade_utils as networks
from spade_utils import util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
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
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt
            )
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

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
        util.save_network(self.netG, "G", epoch, self.opt)
        util.save_network(self.netD, "D", epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, "E", epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, "G", opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, "D", opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, "E", opt.which_epoch, opt)

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

    def __init__(self, model_dir="models/ml4a_spade", model_name="landscapes") -> None:
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
        opt.gpu_ids = []
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

    def get_class_index(self, class_name):
        if class_name not in self.classes.values():
            return None
        return list(self.classes.keys())[list(self.classes.values()).index(class_name)]

    def __call__(self, labelmap):
        labelmap = Image.fromarray(np.array(labelmap).astype(np.uint8))
        params = networks.get_params(self.opt, labelmap.size)
        transform_label = networks.get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False
        )
        label_tensor = transform_label(labelmap) * 255.0
        label_tensor[label_tensor == 255.0] = self.opt.label_nc
        transform_image = networks.get_transform(self.opt, params)
        image_tensor = transform_image(Image.new("RGB", (500, 500)))
        data = {
            "label": label_tensor.unsqueeze(0),
            "instance": label_tensor.unsqueeze(0),
            "image": image_tensor.unsqueeze(0),
        }
        generated = self.model(data, mode="inference")
        output = networks.tensor2im(generated[0])
        output = Image.fromarray(output)
        return output


def main():
    # 载入模型
    spade = Spade()
    semantic_segmentation = SemanticSegmentation()

    # 读取图像
    img_path = "data/semantic_segmentation-samples/mountains.jpeg"
    ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    # 进行分割
    uni_size = (512, 512)
    resized_img = cv2.resize(img, uni_size)
    labels = semantic_segmentation(resized_img)

    # 转换标签号
    labels2 = labels.copy()
    labels2[
        labels == semantic_segmentation.get_class_index("mountain")
    ] = spade.get_class_index("mountain")
    labels2[
        labels == semantic_segmentation.get_class_index("earth")
    ] = spade.get_class_index("grass")
    labels2[
        labels == semantic_segmentation.get_class_index("water")
    ] = spade.get_class_index("river")
    labels2[
        labels == semantic_segmentation.get_class_index("tree")
    ] = spade.get_class_index("tree")
    labels2[
        labels == semantic_segmentation.get_class_index("sky")
    ] = spade.get_class_index("sky-other")

    # 根据labelmap生成图像
    start_time = time.time()
    img2 = spade(labels2)
    print("Cost Time(s):", time.time() - start_time)

    # 可视化
    plt.subplot(1, 2, 1)
    plt.imshow(resized_img)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("SPADE Result")
    plt.axis("off")
    plt.savefig("output/spade.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    # Spade.download_models()
