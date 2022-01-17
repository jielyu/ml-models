# ml-models

To store code for machine learning experiments

## 实验列表 Experiments List

|序号 ID|实验名 Experiment Name |精度 Acc| 框架 Framework| 代码 Code |数据集 Dataset|备注 Mark|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1| vggnet_on_celeba|acc=90.17%|keras|[源码](src/vgg_on_celeba.py)|[CelebA-dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|Without BN|
|2|vggnet_bn_on_celeba|acc=90.62%|keras|[源码](src/vgg_on_celeba.py)|[CelebA-dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|With BN|
|3|smallervgg_bn_on_12306verifycode|acc=95%|keras|[源码](src/smallervgg_on_12306verifycode.py)|[12306verifycode-dataset](.)|With BN|
