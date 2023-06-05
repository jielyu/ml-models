## YOLOX

### 1. YoloX

[模型代码](src/yolox.py)

目标检测模型，包含训练和推理代码，
主要是针对官方代码中网络结构与损失严重耦合造成部署麻烦的问题，
将网络结构与训练技巧分开。

#### colab上训练的指令

```shell
python src/e004_yolox/yolox.py --phase train --gpu 1 --batch_size 8
```

```
python src/e004_yolox/yolox.py --phase train --gpu 1 --batch_size 16 --data_dir /home/data/Dataset
```