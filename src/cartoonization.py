# encoding: utf-8

import cv2
import numpy as np
import tensorflow as tf
import tf_slim as slim

# tf.compat.v1.disable_eager_execution()


def resblock(inputs, out_channel=32, name="resblock"):

    with tf.variable_scope(name):

        x = slim.convolution2d(
            inputs, out_channel, [3, 3], activation_fn=None, scope="conv1"
        )
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(
            x, out_channel, [3, 3], activation_fn=None, scope="conv2"
        )

        return x + inputs


def unet_generator(inputs, channel=32, num_blocks=4, name="generator", reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(x0, channel, [3, 3], stride=2, activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(x1, channel * 2, [3, 3], stride=2, activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(x2, out_channel=channel * 4, name="block_{}".format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)
        x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)

        return x4


def tf_box_filter(x, r):
    k_size = int(2 * r + 1)
    ch = x.get_shape().as_list()[-1]
    weight = 1 / (k_size**2)
    box_kernel = weight * np.ones((k_size, k_size, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], "SAME")
    return output


def guided_filter(x, y, r, eps=1e-2):

    x_shape = tf.shape(x)
    # y_shape = tf.shape(y)

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


class Cartoonization:
    def __init__(
        self, model_path="models/ml4a_cartoonization/White-box-Cartoonization"
    ) -> None:
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024))
        print("start")
        # self.input_photo = tf.compat.v1.placeholder(
        #     tf.float32, shape=(None, None, None, 3)
        # )
        # print("create")
        # network_out = unet_generator(self.input_photo)
        # print("filter")
        # self.final_out = guided_filter(self.input_photo, network_out, r=1, eps=5e-3)
        # print("end")
        # # 创建模型载入器
        # all_vars = tf.compat.v1.trainable_variables()
        # gene_vars = [var for var in all_vars if "generator" in var.name]
        # saver = tf.compat.v1.train.Saver(var_list=gene_vars)
        # # 设置TF的绘画环境
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.compat.v1.Session(config=config)
        # self.sess.run(tf.compat.v1.global_variables_initializer())
        # print("loading..")
        # # 载入模型
        # saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def __call__(self, image):

        # resize so resolution %4==0
        w1, h1 = image.shape[1], image.shape[0]
        w2 = w1 + (4 - w1 % 4)
        h2 = h1 + (4 - h1 % 4)
        img = cv2.resize(image, (w2, h2))
        img = np.array(img)

        # img = resize_crop(img)
        img = img.astype(np.float32) / 127.5 - 1
        if img.ndim < 4:
            img = np.expand_dims(img, axis=0)
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: img})
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (w1, h1))
        return output


def main():
    cart = Cartoonization()


if __name__ == "__main__":
    main()
