from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from KerasLayer import KerasLayer
from mnist_mlp import load_dataset
from keras.callbacks import TensorBoard

global_nb_words = 20000
global_batch_size = 100
global_original_dim = 784
global_intermediate_dim = 256
global_z_dim = 128
global_nb_epochs = 100


class VAE(object):
    def __init__(self,
                 nb_words=global_nb_words,
                 in_dim=global_original_dim,
                 intermediate_dim=global_intermediate_dim,
                 z_dim=global_z_dim,
                 batch_size=global_batch_size,
                 optimizer='rmsprop',
                 debug=False,
                 nb_epochs=global_nb_epochs):
        self.BATCH_SIZE = batch_size
        self.NB_EPOCHS = nb_epochs

        X = Input(batch_shape=(None, in_dim))
        L_enc = Dense(**KerasLayer(n_out=intermediate_dim,
                                   activation='relu')())(X)
        Z_mean = Dense(z_dim)(L_enc)
        Z_stddev = Dense(z_dim)(L_enc)

        def sampling(args):
            z_m, z_stdd = args
            epsilon = K.random_normal(shape=(self.BATCH_SIZE, z_dim),
                                      mean=0.,
                                      std=1.0)
            out = z_m + epsilon * K.exp(z_stdd / 2)  # Element-wise multiplication
            return out

        Z = Lambda(sampling, output_shape=(z_dim,))([Z_mean, Z_stddev])

        L_dec = Dense(**KerasLayer(n_out=intermediate_dim,
                                   activation='relu')())(Z)
        X_out = Dense(**KerasLayer(n_out=in_dim,
                                   activation='sigmoid')())(L_dec)

        def vae_loss(X, X_decoded):
            cross_entropy = objectives.binary_crossentropy(X, X_decoded)  # DIFF from K.binary_crossentropy()
            kl_loss = -0.5 * K.sum(1 + Z_stddev - K.square(Z_mean) - K.exp(Z_stddev), axis=-1)
            return cross_entropy + kl_loss

        vae = Model(X, X_out)

        if debug:
            vae.summary()
        vae.compile(optimizer=optimizer, loss=vae_loss)
        self.vae = vae

    def train(self):
        (X_train, _), (X_test, _) = load_dataset()
        self.vae.fit(X_train, X_train,
                     batch_size=self.BATCH_SIZE,
                     nb_epoch=self.NB_EPOCHS,
                     verbose=2,
                     callbacks=[TensorBoard()],
                     validation_data=(X_test, X_test))

        self.vae.evaluate(X_test, X_test, verbose=0)


if __name__ == '__main__':
    vae = VAE()
    vae.train()
