# ml-models

To store code for machine learning experiments

## 模型介绍

### 1. VggNet

[模型代码](src/vgg_on_celeba.py)
[CelebA-dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

使用VggNet在CelebA数据集上测试分类效果

|序号 ID|实验名 Experiment Name |精度 Acc|备注 Mark|
|:---:|:---:|:---:|:---:|
|1| vggnet_on_celeba|acc=90.17%|Without BN|
|2|vggnet_bn_on_celeba|acc=90.62%|With BN|

### 2. small VggNet

[模型代码](src/smallervgg_on_12306verifycode.py)

使用Small VggNet 对12306的验证码进行识别

|序号 ID|实验名 Experiment Name |精度 Acc|备注 Mark|
|:---:|:---:|:---:|:---:|
|1|smallervgg_bn_on_12306verifycode|acc=95%|With BN|

### 3. YoloX

[模型代码](src/yolox.py)

目标检测模型，包含训练和推理代码，主要是针对官方代码中网络结构与损失严重耦合造成部署麻烦的问题，将网络结构与训练技巧分开。
