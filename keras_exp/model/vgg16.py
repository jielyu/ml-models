from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Input
from keras import backend as K

class Vgg16:
    @staticmethod
    def build(height, width, depth, feat_dim):
        model = Sequential()
        input_shape = (height, width, depth)
        chanDim = -1 # channel lies in the last dim
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # block_1
        model.add(Conv2D(64, (3,3), padding='same'))#, input_shape=input_shape
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # block_2
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # block_3
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # block_4
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # block_5
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # fully-connected layers
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(feat_dim))
        return model


def main():

    inputs = Input(shape=(178, 178, 3))
    vggnet = Vgg16.build(178, 178, 3, 80)
    feat = vggnet(inputs)
    model = Model(inputs=inputs, outputs=feat)
    model.summary()

if __name__ == '__main__':
    main()