# ml-models

分享机器学习实验的代码，为相关方向的人员提供学习资源

## 环境需求

可通过 `requirements.txt` 文件安装所需第三方模块

```shell
pip3 install -r requirements.txt
```

## 数据依赖

以下依赖也可以使用软连接满足。

**注意**：如果出现模型缺失的情况，需要下载 `models` 文件夹，
并放在工程根目录下，即 `models/`

下载地址： [百度网盘](https://pan.baidu.com/s/11Q_QOocEblqGcglH5aJs6A?pwd=9emk)

**注意**：如果出现数据集缺失的情况，需要下载 `dataset` 文件夹，
并放在工程根目录下，即 `dataset/`，并解压对应数据集

下载地址： [百度网盘](https://pan.baidu.com/s/1B1qTWG309_R13fDiGd6mOg?pwd=wuuk)


## 模型介绍

### 14. animestylized

由于官方提供的ckpt依赖特定的工程目录结构，因此无法运行。

想要直接运行，可以查看[官方源码](https://github.com/zhen8838/AnimeStylized)。

也可以从百度网盘下载处理过之后的模型文件。

依赖的vgg模型可以从百度网盘下载后放置到 `models/vggnet/vgg19.npy`

#### animegan

[模型代码](src/animestylized_animegan.py)

测试

```
python src/animestylized_animegan.py --config None --stage infer --ckpt models/animestylized/animeganv2/version_0/checkpoints/epoch=17_no_callbacks.ckpt --extra image_path:data/animestylized-samples/animegan_test2.jpg
```

![animegan效果](docs/images/animestylized_animegan.png)

数据集 

dataset/animestylized/dataset.zip

预训练

```
python src/animestylized_animegan_pretrainpy --config configs/animestylized/animegan_pretrain.yaml --stage fit
```

训练

```
python src/animestylized_animegan.py --config configs/animestylized/animeganv2.yaml --stage fit
```

#### whiteboxgan

[模型代码](src//animestylized_whiteboxgan.py)

测试

```
python src/animestylized_whiteboxgan.py --config None --stage infer --ckpt models/animestylized/whitebox/version_0/checkpoints/epoch=4_no_callbacks.ckpt --extra image_path:data/animestylized-samples//whitebox_test.jpg
```

![whiteboxgan效果](docs/images/animestylized_whiteboxgan.png)

数据集

dataset/animestylized/cvpr_dataset.zip

预训练

```
python src/animestylized_whitebox_pretrainpy --config configs/animestylized/whitebox_pretrain.yaml --stage fit
```

训练

```
python src/animestylized_whiteboxgan.py --config configs/animestylized/whitebox.yaml --stage fit
```

#### uagtit

[模型代码](src/animestylized_uagtit.py)

只能用于处理人脸

测试

```
python src/animestylized_uagtit.py --config None --stage infer --ckpt models/animestylized/uagtit/version_13/checkpoints/epoch=15_no_callbacks.ckpt --extra image_path:data/animestylized-samples/uagtit_test.png
```

![uagtit效果](docs/images/animestylized_uagtit.png)

数据集

dataset/animestylized/cvpr_dataset.zip



### 13. idinvert

[模型代码](src/idinvert.py)

可以修改属性的生成模型

#### 模型依赖

可以从百度网盘下载，并放置到以下路径

`models/ml4a_idinvert`

#### 运行指令

```shell
python src/idinvert.py
```

![融合](docs/images/idinvert-fuse.png)
![生成](docs/images/idinvert-generation.png)
![属性](docs/images/idinvert-attr.png)

### 12. ESRGAN

[模型代码](src/esrgan.py)

生成4x超分辨率图像

#### 模型依赖

可以从百度网盘下载，并放置到以下路径

`models/ml4a_esrgan`

#### 运行指令

```shell
python src/esrgan.py
```

![esrgan](docs/images/esrgan.png)

### 11. Spade

[模型代码](src/spade.py)

根据mask标注信息生成图片

#### 模型依赖

可以从百度网盘下载，并放置到以下路径

`models/ml4a_spade`

#### 运行指令

```shell
python src/spade.py
```

![生成效果](docs/images/spade.png)

### 10. Semantic Segmentation

[模型代码](src/semantic_segmentation.py)

对图片实现语义分割的功能，生成各种语意类别的mask

#### 模型依赖

可以从百度网盘下载，并放置到以下路径

`models/ml4a_semantic_segmentation`

#### 运行指令

```shell
python src/semantic_segmentation.py
```

![语义分割](docs/images/semantic_segmentation.png)

### 9. PhotoSketch

[模型代码](src/photosketch.py)

将照片转换为简笔画

#### 模型依赖

可以从百度网盘下载，并放置到以下路径

`models/ml4a_photosketch/pretrained`

#### 运行指令

```shell
python src/photosketch.py
```

![photosketch](docs/images/photosketch.png)

### 8. BASNet

[模型代码](src/basnet.py)

提取图片中前景物体的mask

#### 模型依赖

可以从百度网盘下载，并放置到以下路径

`models/ml4a_basnet/basnet.pth`

还依赖 `resnet34` 运行时联网会自动从pytorch官网下载

#### 运行指令

```shell
python src/basnet.py
```

![合成图像](docs/images/basnet_final.png)

### 7. dlib Face Recognition

[模型代码](src/dlib_face_recognition.py)

封装和展示dlib中人脸识别模型的效果

```shell
python src/dlib_face_recognition.py
```

![人脸识别](docs/images/dlib_face_recognision.png)

### 6. dlib Face Alignment

[模型代码](src/dlib_face_alignment.py)

封装和展示dlib中人脸对齐模型的效果

```shell
python src/dlib_face_alignment.py
```

![人脸识别](docs/images/dlib_face_alignment.png)

### 5. dlib Face Detection

[模型代码](src/dlib_face_detection.py)

封装和展示dlib中人脸检测器的效果

```shell
python src/dlib_face_detection.py
```

![人脸识别](docs/images/dlib_face_detection.png)

### 4. SVM (pytorch)

[模型代码](src/svm.py)

使用pytorch实现svm的hinge loss，
并用误差反传的迭代方式进行优化，仅供娱乐。

### 3. YoloX

[模型代码](src/yolox.py)

目标检测模型，包含训练和推理代码，
主要是针对官方代码中网络结构与损失严重耦合造成部署麻烦的问题，
将网络结构与训练技巧分开。

#### colab上训练的指令

```shell
python src/yolox.py --phase train --colab 1 --gpu 1 --batch_size 8
```

### 2. small VggNet [Deprecated]

[模型代码](src/smallervgg_on_12306verifycode.py)

github上已经有一个12306验证码识别的[repo](https://github.com/wudinaonao/12306CaptchaCrack)，但是该程序内存的占用实在是有些浪费，以至于在我的小内存机器上无法正常运行，所以对数据集读取进行了一些优化。

#### 训练

```shell
python3 -m keras_exp.train_smallervgg_on_12306verifycode --dataset-dir=path/to/trainval/directory
```

#### 评估

```shell
python3 -m keras_exp.train_smallervgg_on_12306verifycode --phase=evaluate --dataset-dir=path/to/test/directory
```

验证集占比20%， batch_size默认设置为128，训练25个epoch可以达到的精度如下：
train_acc=99.0%, val_acc=99.1%, test_acc=95%

#### 指标

使用Small VggNet 对12306的验证码进行识别

|序号 ID|实验名 Experiment Name |精度 Acc|备注 Mark|
|:---:|:---:|:---:|:---:|
|1|smallervgg_bn_on_12306verifycode|acc=95%|With BN|


### 1. VggNet [Deprecated]

[模型代码](src/vgg_on_celeba.py)
[CelebA-dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### 指标

使用VggNet在CelebA数据集上测试分类效果

|序号 ID|实验名 Experiment Name |精度 Acc|备注 Mark|
|:---:|:---:|:---:|:---:|
|1| vggnet_on_celeba|acc=90.17%|Without BN|
|2|vggnet_bn_on_celeba|acc=90.62%|With BN|

