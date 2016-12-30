from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard


def load_dataset(input_size=784, nb_targets=10):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], input_size)
    X_test = X_test.reshape(X_test.shape[0], input_size)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_targets)
    y_test = np_utils.to_categorical(y_test, nb_targets)

    return (X_train, y_train), (X_test, y_test)


class MLP(object):
    def __init__(self, input_size=784,
                 output_size=10,
                 batch_size=32,
                 nb_epochs=100,
                 activation='relu',
                 dropout=0.3,
                 debug=False):
        self.BATCH_SIZE = batch_size
        self.NB_EPOCHS = nb_epochs
        model = Sequential()
        model.add(Dense(input_size, input_dim=input_size))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        model.add(Dense(output_size))
        model.add(Activation('softmax'))

        if debug:
            model.summary()
        model.compile(optimizer=RMSprop(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train_model(self):
        (X_train, y_train), (X_test, y_test) = load_dataset()
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.BATCH_SIZE,
                                 nb_epoch=self.NB_EPOCHS,
                                 validation_data=(X_test, y_test),
                                 callbacks=TensorBoard(),
                                 verbose=2,  # Full output
                                 )

        score = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test score: %f' % score[0])
        print('Test accuracy: %f' % score[1])


if __name__ == '__main__':
    mlp = MLP()
    mlp.train_model()
