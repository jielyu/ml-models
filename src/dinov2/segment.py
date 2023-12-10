# encoding: utf-8

import faulthandler

faulthandler.enable()

import sys
import math
import itertools
from functools import partial
import urllib

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import mmcv
from mmcv.runner import load_checkpoint

from mmseg.apis import init_segmentor, inference_segmentor

REPO_PATH = "./3rd-part/dinov2"  # Specify a local path to the repository (or use installed package instead)
sys.path.append(REPO_PATH)
import dinov2.eval.segmentation.models
import dinov2.eval.segmentation.utils.colormaps as colormaps


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        output = F.pad(x, pads)
        return output


def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )
    model.init_weights()
    return model


def build_model(
    BACKBONE_SIZE="small", HEAD_SCALE_COUNT=3, HEAD_DATASET="voc2012", HEAD_TYPE="ms"
):
    """创建模型

    Args:
        BACKBONE_SIZE, in ("small", "base", "large" or "giant")
        HEAD_SCALE_COUNT, in (1,2,3,4,5)
        HEAD_DATASET, in ("ade20k", "voc2012")
        HEAD_TYPE, in ("ms, "linear")


    """
    # BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"
    backbone_model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2", model=backbone_name
    )
    backbone_model.eval()
    backbone_model.cuda()

    # HEAD_SCALE_COUNT = 3  # more scales: slower but better results, in (1,2,3,4,5)
    # HEAD_DATASET = "voc2012"  # in ("ade20k", "voc2012")
    # HEAD_TYPE = "ms"  # in ("ms, "linear")

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1][
            "img_ratios"
        ][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

    model = create_segmenter(cfg, backbone_model=backbone_model)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.cuda()
    model.eval()
    return model


def render_segmentation(segmentation_logits, colormap):
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)


def main():
    """入口"""
    # 创建模型
    BACKBONE_SIZE = "small"
    HEAD_SCALE_COUNT = 3
    HEAD_DATASET = "voc2012"
    HEAD_TYPE = "ms"
    model = build_model(BACKBONE_SIZE, HEAD_SCALE_COUNT, HEAD_DATASET, HEAD_TYPE)
    #
    EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    image = load_image_from_url(EXAMPLE_IMAGE_URL)
    # display(image)
    image.save("./output/segment_image.png")
    #
    DATASET_COLORMAPS = {
        "ade20k": colormaps.ADE20K_COLORMAP,
        "voc2012": colormaps.VOC2012_COLORMAP,
    }
    array = np.array(image)[:, :, ::-1]  # BGR
    segmentation_logits = inference_segmentor(model, array)[0]
    segmented_image = render_segmentation(
        segmentation_logits, colormap=DATASET_COLORMAPS[HEAD_DATASET]
    )
    # display(segmented_image)
    segmented_image.save("./output/dinov2_segment.png")


if __name__ == "__main__":
    main()
