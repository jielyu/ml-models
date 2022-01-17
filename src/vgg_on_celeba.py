#encoding: utf-8

import cv2
import pandas as pd
from keras.layers import Input, Reshape, Activation
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Input, Reshape
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.models import load_model, Model
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, \
    ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import os
import time
import random
import argparse
import pickle
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# uncomment if run in terminal
matplotlib.use("Agg")
# uncomment if would like to run on plaidml backend
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


def check_dir(dirname, report_error=False):
    """ Check existence of specific directory
    Args:
        dirname: path to specific directory
        report_error: if it is True, error will be report when dirname not exists
                      if it is False, directory will be created when dirnam not exists
    Return:
        None
    Raise:
        ValueError, if report_error is True and dirname not exists 
    """
    if not os.path.exists(dirname):
        if report_error is True:
            raise ValueError('not exist directory: {}'.format(dirname))
        else:
            os.makedirs(dirname)
            print('not exist {}, but has been created'.format(dirname))


def check_path(path_name):
    if not os.path.exists(path_name):
        raise ValueError('not exist path:{}'.format(path_name))


def get_and_split_dataset(dataset_dir, valid_cols=[]):
    # get path and check existence
    attr_path = os.path.join(dataset_dir, 'Anno', 'list_attr_celeba.txt')
    check_path(attr_path)
    partition_path = os.path.join(
        dataset_dir, 'Eval', 'list_eval_partition.txt')
    check_path(partition_path)
    img_dir = os.path.join(dataset_dir, 'Img', 'img_align_celeba')
    check_path(img_dir)
    # read partition info
    part_df = pd.read_csv(partition_path, sep='\s+', skiprows=0, header=None)
    part_df.columns = ['filename', 'partition']
    part_df = part_df.set_index('filename')
    # part_df.head(5)
    # read attr info
    attr_df = pd.read_csv(attr_path, sep='\s+', skiprows=1)
    attr_df[attr_df == -1] = 0
    # attr_df.head()
    # merge partition and attribution data frame
    df = attr_df.merge(part_df, left_index=True, right_index=True)
    # split into train, val, test partition
    train_df = df.loc[df['partition'] == 0].drop(['partition'], axis=1)
    val_df = df.loc[df['partition'] == 1].drop(['partition'], axis=1)
    test_df = df.loc[df['partition'] == 2].drop(['partition'], axis=1)
    # select chosen attributes
    if valid_cols is not None \
            and isinstance(valid_cols, list) and len(valid_cols) > 0:
        train_df, val_df, test_df = \
            train_df[valid_cols], val_df[valid_cols], test_df[valid_cols]
    return img_dir, train_df, val_df, test_df


def center_crop_index(origin_len, crop_len):
    tmp = int(origin_len/2.0 - crop_len/2.0)
    c_start = tmp if tmp >= 0 else 0
    tmp = c_start + crop_len
    c_end = tmp if tmp <= origin_len else origin_len
    return c_start, c_end


def center_crop(img, crop_size):
    h_start, h_end = center_crop_index(img.shape[0], crop_size[1])
    c_start, c_end = center_crop_index(img.shape[1], crop_size[0])
    return img[h_start:h_end, c_start:c_end, ...]


class Dataset:

    def __init__(self, img_dir, df, batch_size=32,
                 target_shape=(128, 128, 3), shuffle=False):
        self.img_dir = img_dir
        self.img_names = df.index.values
        self.labels = df.values
        self.attrs = df.columns.tolist()
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle

    def get_dataset_size(self):
        return len(self.img_names)

    def get_cate_num(self):
        return 2

    def get_attr_num(self):
        return len(self.attrs)

    def get_attrs(self):
        return self.attrs

    def get_target_shape(self):
        return self.target_shape

    def get_batch_num(self):
        # compute number of batches in an epoch
        num_samples = self.get_dataset_size()
        num_batch = num_samples // self.batch_size
        if num_batch * self.batch_size < num_samples:
            num_batch += 1
        return num_batch

    def generate(self, epoch_stop=False):
        # compute number of batches in an epoch
        num_samples = self.get_dataset_size()
        all_idx = list(range(num_samples))
        num_batch = self.get_batch_num()
        # generate samples batch by batch
        tgt_size = (self.target_shape[1], self.target_shape[0])
        crop_size = (178, 178)
        while True:
            if self.shuffle is True:
                random.shuffle(all_idx)
            for i in range(num_batch):
                #i = 0
                # compute start and end index on all_idx
                start_idx = i * self.batch_size
                tmp_idx = (i+1) * self.batch_size
                end_idx = tmp_idx if tmp_idx <= num_samples else num_samples
                # get a batch of samples
                batch_images = []
                batch_labels = []
                for idx in range(start_idx, end_idx):
                    index = all_idx[idx]
                    # get image path
                    img_name = self.img_names[index]
                    img_path = os.path.join(self.img_dir, img_name)
                    # read image from file by opencv with BGR channels
                    img = cv2.imread(img_path)
                    # center crop
                    img = center_crop(img, crop_size)
                    # resize to uniform size
                    img = cv2.resize(img, tgt_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # normalize
                    img = img / 255.0
                    batch_images.append(img)
                    # get label info
                    batch_labels.append(self.labels[index, ...])
                images = np.array(batch_images, dtype=np.float32)
                labels = np.array(batch_labels)
                # one-hot encoding
                oh_labels = np.zeros(
                    (labels.shape[0], labels.shape[1], self.get_cate_num()), dtype=np.float32)
                for i in range(oh_labels.shape[0]):
                    for j in range(oh_labels.shape[1]):
                        oh_labels[i, j, labels[i, j]] = 1
                yield images, np.squeeze(oh_labels)
            if epoch_stop is True:
                break


def create_dataset(dataset_dir, batch_size=32, target_shape=(128, 128, 3)):
    # get image directory and data frames of datasets
    img_dir, train_df, val_df, test_df = get_and_split_dataset(dataset_dir)
    # create trainset, valset, testset
    trainset = Dataset(img_dir, train_df, batch_size, target_shape, True)
    valset = Dataset(img_dir, val_df, batch_size, target_shape, False)
    testset = Dataset(img_dir, test_df, batch_size, target_shape, False)
    return trainset, valset, testset


def test_dataset():
    # create dataset
    dataset_dir = os.path.expanduser('~/Database/Dataset/CelebA-dataset')
    trainset, valset, testset = create_dataset(dataset_dir)
    print(trainset.get_batch_num())
    print(valset.get_batch_num())
    print(testset.get_batch_num())
    # check generator
    cnt = 0
    for images, labels in trainset.generate(epoch_stop=True):
        cnt += 1
        print(cnt, images.shape, labels.shape)
        print(images.dtype, labels.dtype, labels[-1, ...])
        plt.imshow(images[-1, ...])
        plt.show()
        if cnt >= 10:
            break


class VggNet:
    @staticmethod
    def build(height, width, depth, feat_dim, bn=False, dropout=0.25):
        model = Sequential()
        input_shape = (height, width, depth)
        # could be used in batch normalization
        chan_dim = -1  # channel lies in the last dim
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        # block_1
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_2
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_3
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_4
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_5
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # fully-connected layers
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))

        model.add(Dense(4096))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))

        model.add(Dense(feat_dim))
        return model


def train(args):
    # create dataset
    trainset, valset, _ = create_dataset(
        args.dataset_dir, args.batch_size, TARGET_SHAPE)
    # dump attrs
    with open(args.label_path, 'w') as wfid:
        wfid.write(json.dumps(trainset.get_attrs()))
    # create vggnet
    ishape = trainset.get_target_shape()
    num_attrs, num_cates = trainset.get_attr_num(), trainset.get_cate_num()
    feat_dim = num_attrs * num_cates
    model = VggNet.build(
        ishape[0], ishape[1], ishape[2], feat_dim,
        bn=args.batch_normalize, dropout=args.dropout)
    # reshape to the same shape with labels
    if 1 != num_attrs:
        model.add(Reshape((num_attrs, num_cates)))
    model.add(Activation('softmax'))
    # config parameters of solver
    opt = Adam(lr=args.init_lr)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
    model.summary()
    # create callback
    callbacks = [
        ModelCheckpoint(
            args.model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1),
        # EarlyStopping(patience=50),
        # ReduceLROnPlateau(patience=10),
        CSVLogger(args.trainlog_path)]

    # train model
    if args.goon_train and os.path.exists(args.model_path):
        model.load_weights(args.model_path)
    H = model.fit_generator(
        trainset.generate(),
        steps_per_epoch=trainset.get_batch_num(),
        validation_data=valset.generate(),
        validation_steps=valset.get_batch_num(),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks)
    # plot curve
    plt.style.use("ggplot")
    plt.figure()
    N = args.epochs
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_acc')
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args.plot_path)


def evaluate(args):
    # create dataset
    _, _, testset = create_dataset(
        args.dataset_dir, args.batch_size, TARGET_SHAPE)
    # load model
    model = load_model(args.model_path)
    # evaluate
    ret = model.evaluate_generator(
        testset.generate(),
        steps=testset.get_batch_num(),
        verbose=1
    )
    print('evaluate on testset: loss={}, acc={}'.format(ret[0], ret[1]))
    # for images, labels in testset.generate(epoch_stop=True):
    #     preds = model.predict(images)
    #     print(np.argmax(preds, axis=-1))
    #     print(np.argmax(labels, axis=-1))
    #     break


# default dataset dir
DATASET_HOME = os.path.expanduser('~/Database/Dataset')
DATASET_PATH = os.path.join(DATASET_HOME, 'CelebA-dataset')
# default model path
MODEL_SAVE_PATH = os.path.join('output', 'model', 'vggnet_on_celeba.model')
# default path of label2idx map
LABELBIN_SAVE = os.path.join('output', 'label', 'celeba_attrs.json')
# default curve file path
LOSS_PLOT_PATH = os.path.join('output', 'vgg_celeba_acc_loss.png')
TRAIN_LOG_PATH = os.path.join('output', 'train_vgg_celeba.log.csv')
# default input image shape
TARGET_SHAPE = (128, 128, 3)


def parse_args():
    """ Parse arguments from command line
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset-dir', type=str, default=DATASET_PATH,
                    help="path to input dataset")
    ap.add_argument('-m', '--model-path', type=str, default=MODEL_SAVE_PATH,
                    help="path to output model")
    ap.add_argument('-l', '--label_path', type=str, default=LABELBIN_SAVE,
                    help="path to output label binarizer")
    ap.add_argument('-p', '--plot-path', type=str, default=LOSS_PLOT_PATH,
                    help="path to output accuracy/loss plot")
    ap.add_argument('--trainlog-path', type=str, default=TRAIN_LOG_PATH,
                    help='path to training log')
    ap.add_argument('-b', '--batch-size', type=int, default=64,
                    help='batch size')
    ap.add_argument('-e', '--epochs', type=int, default=20,
                    help='number of epochs')
    ap.add_argument('-i', '--init-lr', type=float, default=1e-5)
    ap.add_argument('--phase', type=str, default='train',
                    choices=['train', 'evaluate'],
                    help='specify operations, train or evaluate')
    ap.add_argument('--goon-train', type=bool, default=False,
                    help='load old model and go on training')
    ap.add_argument('--batch-normalize', type=bool, default=True,
                    help='add batch normalization layers after activations')
    ap.add_argument('--dropout', type=float, default=0.25,
                    help='dropout probability on training')
    args = ap.parse_args()
    # check args
    check_dir(args.dataset_dir, report_error=True)
    check_dir(os.path.dirname(args.model_path))
    check_dir(os.path.dirname(args.label_path))
    check_dir(os.path.dirname(args.plot_path))
    check_dir(os.path.dirname(args.trainlog_path))
    return args


def test_model_structure():
    model = VggNet.build(128, 128, 3, 80, bn=True, dropout=0.25)
    model.add(Reshape((40, 2)))
    model.add(Activation('softmax'))
    model.summary()
    plot_model(model, 'output/vggnet.png', show_shapes=True)


def main():
    args = parse_args()
    if 'train' == args.phase:
        train(args)
    elif 'evaluate' == args.phase:
        evaluate(args)
    else:
        raise ValueError('not allowed phase[{}]'.format(args.phase))


if __name__ == '__main__':
    # test_dataset()
    # test_model_structure()
    main()
