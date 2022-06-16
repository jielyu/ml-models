# ml-models

分享机器学习实验的代码，为相关方向的人员提供学习资源

## 数据依赖

以下依赖也可以使用软连接满足。

**注意**：如果出现模型缺失的情况，需要下载 `models` 文件夹，
并放在工程根目录下，即 `models/`

下载地址： [百度网盘](https://pan.baidu.com/s/1L7kbGovNyLAJIcSZVe8eqA) 提取码: gm3s

**注意**：如果出现数据集缺失的情况，需要下载 `dataset` 文件夹，
并放在工程根目录下，即 `dataset/`，并解压对应数据集

下载地址： [百度网盘](https://pan.baidu.com/s/1DjhpK2TAPGJbhXgoHKLt1Q) 提取码: dg82


## 模型介绍

### 4. SVM (pytorch)

[模型代码](src/svm.py)

使用pytorch实现svm的hinge loss，
并用误差反传的迭代方式进行优化，仅供娱乐。

### 3. YoloX

[模型代码](src/yolox.py)

目标检测模型，包含训练和推理代码，
主要是针对官方代码中网络结构与损失严重耦合造成部署麻烦的问题，
将网络结构与训练技巧分开。

### 2. small VggNet

[模型代码](src/smallervgg_on_12306verifycode.py)

使用Small VggNet 对12306的验证码进行识别

|序号 ID|实验名 Experiment Name |精度 Acc|备注 Mark|
|:---:|:---:|:---:|:---:|
|1|smallervgg_bn_on_12306verifycode|acc=95%|With BN|


### 1. VggNet

[模型代码](src/vgg_on_celeba.py)
[CelebA-dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

使用VggNet在CelebA数据集上测试分类效果

|序号 ID|实验名 Experiment Name |精度 Acc|备注 Mark|
|:---:|:---:|:---:|:---:|
|1| vggnet_on_celeba|acc=90.17%|Without BN|
|2|vggnet_bn_on_celeba|acc=90.62%|With BN|

