# encoding: utf-8

import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


class SVM(nn.Module):
    def __init__(self, input_size, output_size):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        # out = self.sigmoid(out)
        return out

    def get_params(self):
        """获取模型参数"""
        W = self.linear.weight.squeeze().detach().cpu().numpy()
        b = self.linear.bias.squeeze().detach().cpu().numpy()
        return W, b


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.sum(torch.clamp(1 - y * x, min=0))
        return loss


class SVMLoss(nn.Module):
    def __init__(self, gamma=0.01) -> None:
        super(SVMLoss, self).__init__()
        self.hinge_loss = HingeLoss()
        self.gamma = gamma

    def forward(self, x, y, model):
        loss = self.hinge_loss(x, y)
        # 增加L2正则化
        W = model.linear.weight.squeeze()
        loss += self.gamma * torch.sum(W.t() @ W) / 2.0
        # loss += self.gamma * torch.sum(b**2)
        return loss


class Dataset:
    def __init__(self, n_samples=500):
        X, Y = make_blobs(
            n_samples=n_samples, centers=2, random_state=0, cluster_std=0.4
        )
        X = (X - X.mean()) / X.std()
        Y[np.where(Y == 0)] = -1
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        self.X = X
        self.Y = Y
        self.n_samples = len(X)
        self.n_features = len(X[0])
        self.n_classes = len(set(self.Y))

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples

    def visualize(self, W, b):
        """模型划分效果可视化"""
        X = self.X
        delta = 0.001
        x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
        y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
        x, y = np.meshgrid(x, y)
        xy = list(map(np.ravel, [x, y]))

        z = (W.dot(xy) + b).reshape(x.shape)
        z[np.where(z > 1.0)] = 4
        z[np.where((z > 0.0) & (z <= 1.0))] = 3
        z[np.where((z > -1.0) & (z <= 0.0))] = 2
        z[np.where(z <= -1.0)] = 1

        # plt.figure(figsize=(10, 10))
        plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
        plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
        plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
        plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
        plt.tight_layout()
        # plt.show()


class RandomSampler(Sampler):
    """Randomly sample N items from a given list of indices."""

    def __init__(self, size, shuffle, seed):
        self.shuffle = shuffle
        self.seed = seed
        self.size = size
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

    def __iter__(self):
        """Iter."""

        if self.shuffle:
            yield from torch.randperm(self.size, generator=self.g)
        else:
            yield from torch.arange(self.size)

    def __len__(self):
        """Return the number of samples."""
        return self.size


def test_dataset():
    """测试数据集封装的功能"""
    dataset = Dataset()
    print("num_samples={}".format(len(dataset)))
    print("X.shape={}".format(dataset.X.shape))
    print("Y.shape={}".format(dataset.Y.shape))

    sampler = RandomSampler(len(dataset), shuffle=True, seed=0)
    print("sampler.size={}".format(len(sampler)))
    batch_sampler = BatchSampler(sampler, batch_size=10, drop_last=False)
    print("batch_sampler.size={}".format(len(batch_sampler)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
    print("data_loader.size={}".format(len(data_loader)))

    # 迭代数据集
    for idx, batch in enumerate(data_loader):
        print("iter={}, batch[0].shape={}".format(idx, batch[0].shape))
        print("iter={}, batch[1].shape={}".format(idx, batch[1].shape))
        print(
            "iter={}, batch[0][0]{}, batch[1][0]={}".format(
                idx, batch[0][0], batch[1][0]
            )
        )

    # 测试数据集的可视化
    W, b = np.array([[0.5, -0.5]]), 0.5
    plt.figure(1, figsize=(8, 8))
    dataset.visualize(W, b)
    plt.show()


class Experiment:
    def __init__(self, input_dims=2, batch_size=10, lr=0.0001, max_epochs=200) -> None:
        self.input_dims = input_dims
        self.output_dims = 1
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = max_epochs

    def get_model(self, input_dims, output_dims):
        self.model = SVM(input_dims, output_dims)
        return self.model

    def get_train_loader(self, batch_size, shuffle=True):
        dataset = Dataset()
        self.dataset = dataset
        sampler = RandomSampler(len(dataset), shuffle=shuffle, seed=0)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
        return data_loader

    def get_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return self.optimizer

    def get_loss_net(self):
        self.loss_net = SVMLoss()
        return self.loss_net

    def train(self):
        print("start training")
        plt.ion()  # 开启interactive mode 成功的关键函数
        plt.figure(1, figsize=(8, 8))
        # 创建模型
        model = self.get_model(self.input_dims, self.output_dims)
        model.train()
        # 创建损失函数
        loss_net = self.get_loss_net()
        # 创建数据载入器
        train_loader = self.get_train_loader(batch_size=self.batch_size)
        # 创建优化器
        optimizer = self.get_optimizer(self.lr)
        # 开始训练
        loss_vals = []
        for epoch in range(self.epochs):
            for batch_idx, batch in enumerate(train_loader):
                X, y = batch
                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_net(y_pred, y, model)
                loss.backward()
                optimizer.step()
            print("epoch: {}, loss: {}".format(epoch, loss.item()))
            loss_vals.append(loss.item())
            # 绘制SVM的效果图
            W, b = self.model.get_params()
            print("W={}, b={}".format(W, b))
            self.dataset.visualize(W, b)
            plt.title("epoch={}".format(epoch))
            plt.pause(0.01)
        # 训练损失曲线
        plt.figure(1, figsize=(8, 8))
        epochs_arr = np.arange(self.epochs)
        loss_arr = np.array(loss_vals)
        plt.plot(epochs_arr, loss_arr)
        plt.title("loss")
        plt.show()


def main():
    exp = Experiment()
    exp.train()


if __name__ == "__main__":
    main()
    # test_dataset()
