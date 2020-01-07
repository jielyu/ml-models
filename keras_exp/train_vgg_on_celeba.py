import os, time, random, argparse, pickle, json
import numpy as np
import matplotlib
# uncomment if run in terminal
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
# uncomment if would like to run on plaidml backend
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.callbacks import Callback, EarlyStopping, \
    ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import load_model, Model
from keras.layers import Input,Reshape, Activation
from keras_exp.model.vggnet import VggNet

from dataset.celeba import create_dataset
from utils.utils import check_dir

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
    feat_dim =  num_attrs * num_cates 
    model = VggNet.build(ishape[0], ishape[1], ishape[2], feat_dim)
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
    callbacks=[
            ModelCheckpoint(
                args.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1),
            # EarlyStopping(patience=50),
            # ReduceLROnPlateau(patience=10),
            CSVLogger("training.log")]

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
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
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
    ap.add_argument('-b', '--batch-size', type=int, default=64, 
                    help='batch size')
    ap.add_argument('-e', '--epochs', type=int, default=100, 
                    help='number of epochs')
    ap.add_argument('-i', '--init-lr', type=float, default=1e-5)
    ap.add_argument('--phase', type=str, default='train', 
                    choices=['train', 'evaluate'], 
                    help='specify operations, train or evaluate')
    ap.add_argument('--goon-train', type=bool, default=False, 
                    help='load old model and go on training')
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
