# encoding: utf-8

"""
代码参考： https://github.com/zhen8838/AnimeStylized
"""

from typing import Tuple
import torch
from animestylized_utils.animegands import AnimeGANDataModule
from animestylized_utils.common import run_common, log_images
from .animestylized_animegan import AnimeGAN


# 用于预训练生成模型
class AnimeGANPreTrain(AnimeGAN):
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx):
        input_photo = batch[0]

        generated = self.generator(input_photo)

        real_feature_map = self.pretrained(input_photo)
        fake_feature_map = self.pretrained(generated)
        init_c_loss = self.l1_loss(real_feature_map, fake_feature_map)
        loss = self.hparams.con_weight * init_c_loss

        self.log_dict({"loss": loss})

        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.5, 0.999)
        )
        return [opt_g], []


if __name__ == "__main__":
    run_common(AnimeGANPreTrain, AnimeGANDataModule)
