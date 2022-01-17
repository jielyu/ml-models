# encoding: utf-8

import os
import time
import random
import argparse
import pickle
import json
from collections import defaultdict
import cv2
from imutils import paths
import numpy as np
import matplotlib
# uncomment if run in terminal
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# uncomment if would like to run on plaidml backend
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.utils import plot_model


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


class SmallerVGGNet():

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1  # channel lies in the last dim

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # block_1
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        # block_2
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # block_3
        model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # block_4
        model.add(Conv2D(256, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # block_5
        model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # fully-connected layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # predict probabilities for all categories
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


def get_label_name(image_path):
    """ Get label name from image path
    """
    items = image_path.split(os.path.sep)
    if len(items) < 2:
        raise ValueError('invalid image path:{}'.format(image_path))
    return items[-2]


def get_and_split_dataset(dataset_dir, test_size=0.2):
    """ Get all data and split it into trainset and testset
    Args:
        dataset_dir: path to trainval directory
        test_size: ratio of testset samples over all samples
    Return:
        train_img_paths, list of image paths, None if test_size>=1.0
        test_img_paths,  list pf image paths, None if test_size<=0.0
        label2idx, defaultdict, a map from label_name to index
        idx2label, defaultdict, a map from index to label_name
    """
    # get paths of all samples
    image_paths = sorted(list(paths.list_images(dataset_dir)))
    random.seed(42)
    random.shuffle(image_paths)
    # get labels
    label_dict = defaultdict(int)
    for idx, image_path in enumerate(image_paths):
        label_name = get_label_name(image_path)
        label_dict[label_name] += 1
    label2idx = defaultdict(int)
    idx2label = defaultdict(str)
    num_samples = len(image_paths)
    print('distribution of all categories in this dataset as following:')
    for idx, label_name in enumerate(label_dict.keys()):
        num = label_dict[label_name]
        print('{} cate includes {}/{} samples, {}%'.format(
            label_name, num, num_samples,
            float(num) * 100 / num_samples))
        label2idx[label_name] = idx
        idx2label[idx] = label_name
    # split trainset and testset
    if test_size <= 0.0:
        train_img_paths = image_paths
        test_img_paths = None
    elif test_size >= 1.0:
        train_img_paths = None
        test_img_paths = image_paths
    else:
        bound_idx = int(num_samples * (1.0 - test_size))
        train_img_paths = image_paths[:bound_idx]
        test_img_paths = image_paths[bound_idx:]
    return train_img_paths, test_img_paths, label2idx, idx2label


class Dataset:
    """ To provide interface to get sample batch by batch for training and evaluating
    """

    def __init__(self,
                 image_paths,
                 label2idx,
                 idx2label,
                 target_shape=(67, 67, 3),
                 batch_size=32,
                 shuffle=False):
        self.image_paths = image_paths
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataset_size(self):
        """Get number of all samples on current dataset
        """
        return len(self.image_paths)

    def get_num_cates(self):
        """Get number of categories on current task
        """
        return len(self.label2idx)

    def generate(self, augment=None, epoch_stop=False):
        """Get a generator to orgnize sample batch by batch
        Args:
            augment, an object contains augmentation operations
            epoch_stop, if True, iteration will be stopped in the end of the first epoch
        """
        # compute number of batchs in an epoch
        num_samples = self.get_dataset_size()
        num_batch = num_samples // self.batch_size
        if num_batch * self.batch_size < num_samples:
            num_batch += 1
        while True:
            if self.shuffle is True:
                random.shuffle(self.image_paths)
            # produce all batches
            for i in range(num_batch):
                # compute start and end index
                start_idx = i * self.batch_size
                tmp_idx = (i + 1) * self.batch_size
                end_idx = tmp_idx if tmp_idx <= num_samples else num_samples
                # get a batch of image paths
                batch_img_paths = self.image_paths[start_idx:end_idx]
                # read image and get label
                batch_images = []
                batch_labels = []
                for image_path in batch_img_paths:
                    image = cv2.imread(image_path)
                    image = cv2.resize(
                        image, (self.target_shape[1], self.target_shape[0]))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = img_to_array(image) / 255.0
                    batch_images.append(image)
                    batch_labels.append(
                        self.label2idx[get_label_name(image_path)])
                # normalize image data
                images = np.array(batch_images, dtype='float')
                labels = np.array(batch_labels)
                # return generator
                yield images, np.eye(self.get_num_cates(),
                                     dtype='float')[labels]
            if epoch_stop is True:
                break


def train(args):
    """ Train model on trainset and validate on valset
    """
    # get image paths and split dataset
    trainval_dir = os.path.join(args.dataset_dir, 'train_val')
    check_dir(trainval_dir, report_error=True)
    train_img_paths, test_img_paths, label2idx, idx2label = \
        get_and_split_dataset(trainval_dir, test_size=args.test_ratio)
    with open(args.label_path, 'w') as wfid:
        wfid.write(json.dumps(label2idx))

    # create augumentation operations
    aug = ImageDataGenerator(rotation_range=25,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")

    # create generators of trainset and testset
    trainset = Dataset(image_paths=train_img_paths,
                       label2idx=label2idx,
                       idx2label=idx2label,
                       target_shape=TARGET_SHAPE,
                       batch_size=args.batch_size,
                       shuffle=True)
    valset = Dataset(image_paths=test_img_paths,
                     label2idx=label2idx,
                     idx2label=idx2label,
                     target_shape=TARGET_SHAPE,
                     batch_size=args.batch_size,
                     shuffle=False)
    # build model
    model = SmallerVGGNet.build(width=TARGET_SHAPE[1],
                                height=TARGET_SHAPE[0],
                                depth=TARGET_SHAPE[2],
                                classes=len(label2idx))
    opt = Adam(lr=args.init_lr, decay=args.init_lr / args.epochs)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    model.summary()

    # create callback
    callbacks = [
        ModelCheckpoint(args.model_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1),
        # EarlyStopping(patience=50),
        ReduceLROnPlateau(patience=10),
        CSVLogger("training.log")
    ]

    # train model
    H = model.fit_generator(
        trainset.generate(),
        steps_per_epoch=trainset.get_dataset_size() // args.batch_size,
        validation_data=valset.generate(),
        validation_steps=valset.get_dataset_size() // args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks)

    # plot curve
    plt.style.use("ggplot")
    plt.figure()
    N = args.epochs
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args.plot_path)


def evaluate(args):
    """ Evaluate model on testset
    """
    # get image paths and split dataset
    testset_dir = os.path.join(args.dataset_dir, 'test')
    check_dir(testset_dir, report_error=True)
    _, test_img_paths, _, _ = \
        get_and_split_dataset(testset_dir, test_size=1.0)
    with open(args.label_path, 'r') as fid:
        label2idx = json.loads(fid.read())
    idx2label = defaultdict(str)
    for k, v in label2idx.items():
        idx2label[v] = k
    # create dataset
    testset = Dataset(test_img_paths,
                      label2idx=label2idx,
                      idx2label=idx2label,
                      target_shape=TARGET_SHAPE,
                      batch_size=args.batch_size,
                      shuffle=False)

    # load model
    model = load_model(args.model_path)

    # get prediction
    error_cnt = 0
    for images, labels in testset.generate(epoch_stop=True):
        # predict
        gt_labels = np.argmax(labels, axis=1)
        pred_probs = model.predict(images)
        print(pred_probs.shape)
        pred_labels = np.argmax(pred_probs, axis=1)
        # print results
        for idx, gt in enumerate(gt_labels):
            pred = pred_labels[idx]
            print('gt={}, pred={}'.format(idx2label[gt], idx2label[pred]))
            true_or_false = True
            if gt != pred:
                error_cnt += 1
                true_or_false = False
            if args.save_samples is True:
                img_path = 'output/test_{}_gt_{}-pred_{}.png'.format(
                    true_or_false, idx2label[gt], idx2label[pred])
                img = images[idx, ...] * 255
                img = img.astype(np.uint8)
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    num_samples = testset.get_dataset_size()
    print('total acc={}%'.format(
        (num_samples - error_cnt) * 100 / float(num_samples)))


# default dataset dir
DATASET_HOME = os.path.expanduser('~/Database/Dataset')
DATASET_PATH = os.path.join(DATASET_HOME, '12306verifycode-dataset')
# default model path
MODEL_SAVE_PATH = os.path.join('output', 'model', '12306verifycode.model')
# default path of label2idx map
LABELBIN_SAVE = os.path.join('output', 'label', '12306cate.json')
# default curve file path
LOSS_PLOT_PATH = os.path.join('output', 'accuracy_and_loss.png')
# default input image shape
TARGET_SHAPE = (67, 67, 3)


def parse_args():
    """ Parse arguments from command line
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-d',
                    '--dataset-dir',
                    type=str,
                    default=DATASET_PATH,
                    help="path to input dataset")
    ap.add_argument('-t',
                    '--test-ratio',
                    type=float,
                    default=0.2,
                    help='ratio of samples in testset over the whole dataset')
    ap.add_argument('-m',
                    '--model-path',
                    type=str,
                    default=MODEL_SAVE_PATH,
                    help="path to output model")
    ap.add_argument('-l',
                    '--label_path',
                    type=str,
                    default=LABELBIN_SAVE,
                    help="path to output label binarizer")
    ap.add_argument('-p',
                    '--plot-path',
                    type=str,
                    default=LOSS_PLOT_PATH,
                    help="path to output accuracy/loss plot")
    ap.add_argument('-b',
                    '--batch-size',
                    type=int,
                    default=128,
                    help='batch size')
    ap.add_argument('-e', '--epochs', type=int, default=25, help='')
    ap.add_argument('-i', '--init-lr', type=float, default=1e-3)
    ap.add_argument('--phase',
                    type=str,
                    default='train',
                    choices=['train', 'evaluate'],
                    help='specify operations, train or evaluate')
    ap.add_argument('--save-samples',
                    type=bool,
                    default=True,
                    help='flag to indicate whether save samples on evaluating')
    args = ap.parse_args()
    # check args
    check_dir(args.dataset_dir, report_error=True)
    check_dir(os.path.dirname(args.model_path))
    check_dir(os.path.dirname(args.label_path))
    check_dir(os.path.dirname(args.plot_path))
    return args


def test_model_structure():
    model = SmallerVGGNet.build(67, 67, 3, 80)
    model.summary()
    plot_model(model, 'output/smallerbggnet.png', show_shapes=True)


def main():
    args = parse_args()
    if 'train' == args.phase:
        train(args)
    elif 'evaluate' == args.phase:
        evaluate(args)
    else:
        raise ValueError('not allowed phase[{}]'.format(args.phase))


if __name__ == '__main__':
    # test_model_structure()
    main()
