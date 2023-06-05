# encoding: utf-8

from yolox_utils.mscoco import COCODataset
from yolox_utils.mosaic_detection import MosaicDetection
from yolox_utils.data_augment import TrainTransform


def main():
    # 创建MSCOCO数据集封装
    dataset = COCODataset(
        data_dir="/home/data/Dataset",
        json_file="instances_train2017.json",
        img_size=(640, 640),
        preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        cache=False,
    )
    # 添加样本增强
    dataset = MosaicDetection(
        dataset,
        mosaic=True,
        img_size=(640, 640),
        preproc=TrainTransform(max_labels=120, flip_prob=0.5, hsv_prob=1.0),
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.1, 2),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )
    sample = dataset[0]
    print(len(sample), sample[2])


if __name__ == "__main__":
    main()
