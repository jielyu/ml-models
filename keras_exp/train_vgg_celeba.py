import os, time, random, argparse, pickle, json
import matplotlib
# uncomment if run in terminal
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
# uncomment if would like to run on plaidml backend
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import load_model, Model
from keras.layers import Input,Reshape, Activation
from keras_exp.model.vgg16 import Vgg16

from dataset.celeba import create_dataset

def train(args):
    # create dataset
    trainset, valset, testset = create_dataset(args.dataset_dir, args.batch_size, TARGET_SHAPE)
    # create vggnet
    ishape = trainset.get_target_shape()
    num_attrs, num_cates = trainset.get_attr_num(), trainset.get_cate_num()
    feat_dim =  num_attrs * num_cates 
    vggnet = Vgg16.build(ishape[0], ishape[1], ishape[2], feat_dim)
    # extract features from vggnet
    input_x = Input(shape=ishape)
    feat = vggnet(input_x)
    # reshape to the same shape with labels
    if 1 == num_attrs:
        preds = feat
    else:
        preds = Reshape((num_attrs, num_cates))(feat)
    outputs = Activation('softmax')(preds)
    # construct model and config parameters of solver 
    model = Model(inputs=input_x, outputs=outputs)
    opt = Adam(lr=args.init_lr, decay=args.init_lr / args.epochs)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
    model.summary()
    # create callback
    callbacks=[
            ModelCheckpoint(
                args.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1),
            # EarlyStopping(patience=50),
            ReduceLROnPlateau(patience=10),
            CSVLogger("training.log")]

    # train model
    H = model.fit_generator(
        trainset.generate(),
        steps_per_epoch=trainset.get_dataset_size() // args.batch_size,
        validation_data=valset.generate(),
        validation_steps=valset.get_dataset_size() // args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks)


def evaluate(args):
    pass


# default dataset dir
DATASET_HOME = os.path.expanduser('~/Database/Dataset')
DATASET_PATH = os.path.join(DATASET_HOME, 'CelebA-dataset')
# default model path
MODEL_SAVE_PATH = os.path.join('output', 'model', 'celeba.model')
# default path of label2idx map
LABELBIN_SAVE = os.path.join('output', 'label', 'celeba.json')
# default curve file path
LOSS_PLOT_PATH = os.path.join('output', 'accuracy_and_loss.png')   
# default input image shape
TARGET_SHAPE = (128, 128, 3)
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
    ap.add_argument('-b', '--batch-size', type=int, default=64, 
                    help='batch size')
    ap.add_argument('-e', '--epochs', type=int, default=100, 
                    help='number of epochs')
    ap.add_argument('-i', '--init-lr', type=float, default=1e-3)
    ap.add_argument('--phase', type=str, default='train', 
                    choices=['train', 'evaluate'], 
                    help='specify operations, train or evaluate')
    args = ap.parse_args()
    # check args
    check_dir(args.dataset_dir, report_error=True)
    check_dir(os.path.dirname(args.model_path))
    check_dir(os.path.dirname(args.label_path))
    check_dir(os.path.dirname(args.plot_path))
    return args


def main():
    args = parse_args()
    if 'train' == args.phase:
        train(args)
    elif 'evaluate' == args.phase:
        evaluate(args)
    else:
        raise ValueError('not allowed phase[{}]'.format(args.phase))

        
if __name__ == '__main__':
    main()
