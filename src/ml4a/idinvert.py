# encoding: utf-8

"""
参考代码： https://github.com/ml4a/ml4a
参考代码： https://github.com/genforce/idinvert_pytorch
"""

import sys

sys.path.append("src/e003_dlib/")

import os
from random import sample
import cv2
from matplotlib import pyplot as plt

import numpy as np
from common_utils.download_utils import download_from_gdrive
from common_utils.face_utils import FaceUtils
from dlib_face_detection import FaceDetector
from idinvert_utils.inverter import StyleGANInverter
from idinvert_utils.helper import build_generator


class IdInvert:

    models_config = {
        "bedroom": {
            "name": "styleganinv_bedroom256",
            "inverter": "1ebuiaQ7xI99a6ZrHbxzGApEFCu0h0X2s",
            "generator": "1ka583QwvMOtcFZJcu29ee8ykZdyOCcMS",
            "attributes": {
                "cloth": "1PiOFd71eYTrJclwptYyxqbLBhUSQ0obT",
                "cluttered_space": "1RBWZKE_NlI2cj4aG50VBVAFQZjthsEwL",
                "indoor_lighting": "1z-egLTDGgJsHWiqCO2bQgFW4aHv2iYf5",
                "scary": "1Bc19lhx4MQ_E9vGB02GRaFX7NpeNYPgd",
                "soothing": "1s5vjjo3QbCYphaMOjsMwOLp9bzyO8S2E",
                "wood": "1qOm1QehLJAeH2EQAPmufiVnmz7RWK-WN",
            },
        },
        "ffhq": {
            "name": "styleganinv_ffhq256",
            "inverter": "1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO",
            "generator": "1SjWD4slw612z2cXa3-n38JwKZXqDUerG",
            "attributes": {
                "age": "1ez85GdHz9HZ6DgdQLMmji3ygdMoPipC-",
                "expression": "1XJHe2gQKJEczBEhu2MGA94EX28AssnKM",
                "eyeglasses": "1fFsNwMUUaPq_Hh6uPgA5K-v9Yjq8unjZ",
                "gender": "1iWPlPYHl5h2UsB_ojqB8udJKqvn4y38w",
                "pose": "1WSinkKoX9Y8xzfM0Ff2I6Jdum_nFgAy1",
            },
        },
        "tower": {
            "name": "styleganinv_tower256",
            "inverter": "1Pzkgdi3xctdsCZa9lcb7dziA_UMIswyS",
            "generator": "1lI_OA_aN4-O3mXEPQ1Nv-6tdg_3UWcyN",
            "attributes": {
                "clouds": "18awC-Nq2Anx6qR-Kl2hteFxhQoo7vT9c",
                "sunny": "1dZIG2UoXEszzySh1PP80Dlfi9XQJVeNJ",
                "vegetation": "1LjhoneQ7vTXQ8lJb_CZeTF85ymtfwviB",
            },
        },
    }

    @staticmethod
    def download_models():

        for model_name, model_info in IdInvert.models_config.items():
            # 下载编码器
            model_dir = "ml4a_idinvert"
            encoder_path = os.path.join(model_dir, model_info["name"] + "_encoder.pth")
            encoder_key = model_info["inverter"]
            download_from_gdrive(encoder_key, encoder_path)
            # 下载生成器
            decoder_path = os.path.join(
                model_dir, model_info["name"] + "_generator.pth"
            )
            decoder_key = model_info["generator"]
            download_from_gdrive(decoder_key, decoder_path)
            # 下载属性模型
            for attribute_name, attribute_key in model_info["attributes"].items():
                attribute_path = os.path.join(
                    model_dir, "attributes_{}/{}.npy".format(model_name, attribute_name)
                )
                download_from_gdrive(attribute_key, attribute_path)
        # 下载vgg模型
        download_from_gdrive(
            "1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y", os.path.join(model_dir, "vgg16.pth")
        )

    def __init__(
        self, model_name, num_iterations=1000, regularization_loss_weight=2
    ) -> None:
        if model_name not in IdInvert.models_config:
            raise ValueError("Unknown model name: {}".format(model_name))
        self.model_name = model_name
        model_info = IdInvert.models_config[model_name]
        # 加载编码器
        model_dir = "models/ml4a_idinvert"
        encoder_path = os.path.join(model_dir, model_name + "_encoder.pth")
        self.inverter = StyleGANInverter(
            model_info["name"],
            learning_rate=0.01,
            iteration=num_iterations,
            reconstruction_loss_weight=1.0,
            perceptual_loss_weight=5e-5,
            regularization_loss_weight=regularization_loss_weight,
        )

        # 加载生成器
        decoder_path = os.path.join(model_dir, model_name + "_generator.pth")
        self.generator = build_generator(model_info["name"])
        # 加载属性模型
        self.attributes = {}
        for attribute_name, attribute_key in model_info["attributes"].items():
            attribute_path = os.path.join(
                model_dir, "attributes_{}/{}.npy".format(model_name, attribute_name)
            )
            boundary_filename = attribute_path
            boundary_file = np.load(boundary_filename, allow_pickle=True)[()]
            boundary = boundary_file["boundary"]
            manipulate_layers = boundary_file["meta_data"]["manipulate_layers"]
            self.attributes[attribute_name] = [boundary, manipulate_layers]

        print("加载模型完毕")

    def invert(self, img):
        """编码图像"""
        latent_code, reconstruction = self.inverter.easy_invert(img, num_viz=1)
        return latent_code

    def generate(self, latent_code):
        """生成图像"""
        return self.generator.easy_synthesize(
            latent_code, **{"latent_space_type": "wp"}
        )["image"]

    def modulate(self, latent_code, attr_name, amount):
        """修改属性"""
        if attr_name not in self.attributes:
            raise ValueError("Unknown attribute name: {}".format(attr_name))
        new_code = latent_code.copy()
        manipulate_layers = self.attributes[attr_name][1]
        new_code[:, manipulate_layers, :] += (
            self.attributes[attr_name][0][:, manipulate_layers, :] * amount
        )
        return new_code

    def fuse(
        self, context_images, target_image, crop_size=125, center_x=145, center_y=125
    ):
        """融合"""
        target_image = np.array(target_image)
        top = center_y - crop_size // 2
        left = center_x - crop_size // 2
        width, height = crop_size, crop_size

        if np.array(context_images).ndim < 4:
            context_images = [context_images]

        showed_fuses = []
        resolution = 256
        for context_image in context_images:
            mask_aug = np.ones((resolution, resolution, 1), np.uint8) * 255
            paste_image = np.array(context_image).copy()
            paste_image[top : top + height, left : left + width] = target_image[
                top : top + height, left : left + width
            ].copy()
            showed_fuse = np.concatenate([paste_image, mask_aug], axis=2)
            showed_fuses.append(showed_fuse)

        _, diffused_images = self.inverter.easy_diffuse(
            target=target_image,
            context=np.array(context_images),
            center_x=center_x,
            center_y=center_y,
            crop_x=width,
            crop_y=height,
            num_viz=1,
        )

        diffused_images = [
            np.concatenate([images[-1], mask_aug], axis=2)
            for key, images in diffused_images.items()
        ]

        return showed_fuses, diffused_images


def main():
    model = IdInvert("ffhq", num_iterations=5)
    face_utils = FaceUtils()
    sample_paths = [
        "data/idinvert-samples/1200px-Vladimir_Putin_(2020-02-20).jpg",
        "data/idinvert-samples/Angela_Merkel_2019_(cropped).jpg",
        "data/idinvert-samples/potus2_0.jpg",
    ]
    # 载入原图，并获取对齐图
    img_list, aligned_list = [], []
    plt.figure()
    name_list = ["putin", "merkel", "trump"]
    for idx, p in enumerate(sample_paths):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
        aligned_img = face_utils.align(img)
        aligned_list.append(aligned_img)
        plt.subplot(1, len(sample_paths), idx + 1)
        plt.imshow(aligned_list[idx])
        plt.axis("off")
        plt.title(name_list[idx])
    # plt.show()

    ctx_imgs = aligned_list[1:]
    target_img = aligned_list[0]

    # 人脸融合

    fused_imgs, diffused_imgs = model.fuse(ctx_imgs, target_img)
    plt.figure()
    name_list = ["merkel+putin", "trump+putin"]
    for idx, img in enumerate(diffused_imgs):
        plt.subplot(1, len(diffused_imgs), idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(name_list[idx])
    plt.savefig("output/idinvert-fuse.png", bbox_inches="tight", dpi=300)
    # plt.show()

    # 编码
    latent_code = model.invert(target_img)
    new_img = model.generate(latent_code)[0, :]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(target_img)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(new_img)
    plt.axis("off")
    plt.title("Generation Image")
    plt.savefig("output/idinvert-generation.png", bbox_inches="tight", dpi=300)
    # plt.show()

    # 操作属性
    young = model.modulate(latent_code, "age", -2.0)
    old = model.modulate(latent_code, "age", 2.0)
    yound_img = model.generate(young)[0, :]
    old_img = model.generate(old)[0, :]
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(target_img)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(yound_img)
    plt.axis("off")
    plt.title("Young Image")
    plt.subplot(1, 3, 3)
    plt.imshow(old_img)
    plt.axis("off")
    plt.title("Old Image")
    plt.savefig("output/idinvert-attr.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    # IdInvert.download_models()
