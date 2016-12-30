class KerasLayer(object):
    """
    Own implementation of a Keras layer for standart feedforward networks.

    """

    def __init__(self,
                 n_out,
                 init='glorot_uniform',
                 activation=None,
                 W_regularizer=None,
                 b_regularizer=None):
        self.dic = {'activation': activation,
                    'init': init,
                    'output_dim': n_out,
                    'W_regularizer': W_regularizer,
                    'b_regularizer': b_regularizer}

    def __call__(self, *args, **kwargs):
        return self.dic
