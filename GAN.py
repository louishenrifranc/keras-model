from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Convolution2D, \
    LeakyReLU, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import random

## keras.backend.image_dim_ordering()
# WARNING: dim_ordering = 'th' in the code!!

debug = 1


def load_dataset(input_size=784, nb_targets=10):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

    X_train /= 255
    X_test /= 255
    y_train = np_utils.to_categorical(y_train, nb_targets)
    y_test = np_utils.to_categorical(y_test, nb_targets)

    return (X_train, y_train), (X_test, y_test)


def build_generator():
    input = Input(name='input_generator', shape=[100])
    H = Dense(name='dense_gen1', output_dim=200 * 14 * 14)(input)
    H = BatchNormalization(name='batch_gen1', mode=2)(H)
    H = Activation('relu')(H)
    H = Reshape([200, 14, 14])(H)  # (200, 14, 14)
    H = UpSampling2D(size=(2, 2))(H)  # (200, 28, 28)
    # If stride = 2, and if PADDING=SAME
    # each side padd with floor(stride / 2)
    H = Convolution2D(nb_filter=100, border_mode='same', nb_row=3, nb_col=3, subsample=(1, 1))(H)  # (100,28, 28)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(nb_filter=25, border_mode='same', nb_row=3, nb_col=3, subsample=(1, 1))(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(nb_filter=1, border_mode='same', nb_row=1, nb_col=1)(H)
    H = Activation('sigmoid')(H)

    generator = Model(input, H)
    generator.compile(loss='binary_crossentropy', optimizer=Adam(1e-4))
    if debug: generator.summary()
    return generator


def build_discriminator():
    input = Input(name='input_discriminator', shape=(1, 28, 28))  # Tchecker la différence avec batch_shape
    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.2)(H)
    H = Convolution2D(nb_filter=512, border_mode='same', nb_row=5, nb_col=5, subsample=(2, 2), activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.2)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.2)(H)
    H = Dense(2, activation='softmax')(H)
    # 2 output:
    # first = 1 if true image
    # second = 1 if generated image
    discriminator = Model(input, H)
    discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3))
    if debug: discriminator.summary()
    return discriminator


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def build_GAN():
    generator = build_generator()
    discriminator = build_discriminator()

    gan_input = Input(shape=[100])
    H = generator(gan_input)
    gan_V = discriminator(H)
    GAN = Model(gan_input, gan_V)
    GAN.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4))
    if debug: GAN.summary()
    return GAN, discriminator, generator


def train():
    (X_train, y_train), (X_test, y_test) = load_dataset()
    GAN, discriminator, generator = build_GAN()

    # PRE train the discriminator work
    make_trainable(discriminator, False)

    nb_train_ex = 10000
    train_idx = random.sample(range(0, X_train.shape[0]), nb_train_ex)
    noise_gen = np.random.uniform(0, 1, (nb_train_ex, 100))
    generated_images = generator.predict(noise_gen)

    X = np.concatenate((generated_images, X_train[train_idx, :, :, :]))
    y = np.zeros((2 * nb_train_ex, 2))
    y[:nb_train_ex, 1] = 1
    y[nb_train_ex:, 0] = 1

    make_trainable(discriminator, True)
    discriminator.fit(X, y, nb_epoch=1, batch_size=64)
    y_hat = discriminator.predict(X)
    y_hat = np.argmax(y_hat, axis=1)
    y = np.argmax(y, axis=1)
    err = np.sum(y != y_hat)
    print('Accuracy %f' % (err / y_hat.shape[0]))


if __name__ == '__main__':
    train()
