# encoding: utf-8

"""
用于解决pytorch lightning保存的ckpt文件包含callback导致无法加载的问题
"""

import os
import torch


def remove_callbacks(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise ValueError("not exist file: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    del ckpt["callbacks"]
    fn, ext = os.path.splitext(ckpt_path)
    torch.save(ckpt, fn + "_no_callbacks" + ext)
    print("save to", fn + "_no_callbacks" + ext)


def main():
    root_dir = "root/dir/of/models"
    model_list = [
        "animeganv2/version_0/checkpoints/epoch=17.ckpt",
        "uagtit/version_13/checkpoints/epoch=15.ckpt",
        "whitebox/version_0/checkpoints/epoch=4.ckpt",
        "whitebox/version_1/checkpoints/epoch=4.ckpt",
        "whitebox/version_2/checkpoints/epoch=6.ckpt",
    ]
    for model_file in model_list:
        model_path = os.path.join(root_dir, model_file)
        try:
            remove_callbacks(model_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
