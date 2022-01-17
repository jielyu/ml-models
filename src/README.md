# Introduction of Model Experiments 模型实验说明

## 12306verifycode

[源码](src/smallervgg_on_12306verifycode.py)

github上已经有一个12306验证码识别的[repo](https://github.com/wudinaonao/12306CaptchaCrack)，但是该程序内存的占用实在是有些浪费，以至于在我的小内存机器上无法正常运行，所以对数据集读取进行了一些优化。

### 数据集

可以直接从[repo](https://github.com/wudinaonao/12306CaptchaCrack)中获取数据集，不过该数据测试集目录的组织与训练集有些不一致，需要进行调整。也可以发送关键字【数据】到“青衣极客”公众号获取下载链接，该数据已经调整测试集目录，可以直接使用。

百度网盘中数据集与[repo](https://github.com/wudinaonao/12306CaptchaCrack)的数据集是一致的，只是test_data中文件的组织形式改成与训练集一致。

本工程中的smallervggnet也是直接来源于该[repo](https://github.com/wudinaonao/12306CaptchaCrack)。
如果原作者觉得这种方式存在侵权，可以联系本人删掉网盘共享的数据集。

### 环境需求

    keras
    opencv-python
    imutils
    matplotlib
    numpy

可通过requirements.txt文件安装所需第三方模块

```shell
pip3 install -r requirements.txt
```

### 训练

```shell
python3 -m keras_exp.train_smallervgg_on_12306verifycode --dataset-dir=path/to/trainval/directory
```

如果出现显卡的显存不足，可以适当降低batch_size

### 评估

```shell
python3 -m keras_exp.train_smallervgg_on_12306verifycode --phase=evaluate --dataset-dir=path/to/test/directory
```

验证集占比20%， batch_size默认设置为128，训练25个epoch可以达到的精度如下：
train_acc=99.0%, val_acc=99.1%, test_acc=95%

### 说明

暂时尚未加入"数据增强"的操作

## VGG on Celeba

[源码](src/vgg_on_celeba.py)
