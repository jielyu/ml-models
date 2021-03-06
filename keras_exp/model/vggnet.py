from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Input, Reshape
from keras import backend as K
from keras.utils import plot_model

class VggNet:
    @staticmethod
    def build(height, width, depth, feat_dim, bn=False, dropout=0.25):
        model = Sequential()
        input_shape = (height, width, depth)
        # could be used in batch normalization
        chan_dim = -1 # channel lies in the last dim
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        # block_1
        model.add(Conv2D(64, (3,3), padding='same', input_shape=input_shape))#
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_2
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_3
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_4
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))
        # block_5
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        if bn is True:
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
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


def main():
    model = VggNet.build(128, 128, 3, 80, bn=True, dropout=0.25)
    model.add(Reshape((40, 2)))
    model.add(Activation('softmax'))
    model.summary()
    plot_model(model, 'output/vggnet.png', show_shapes=True)

if __name__ == '__main__':
    main()
