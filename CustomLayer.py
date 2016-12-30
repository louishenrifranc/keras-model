"""

Formula
-------
h_* = u dot v
h_+ = || u - v ||

h = tanh(W_* dot h_*  + W_+ dot h_+)
y = softmax(U dot h + c)

Dimensions
----------
u:  [d, ]
W_: [d_h, d]
U : [ 1, d_h]
"""

from keras.layers import Layer, Dense, Embedding, Permute  # For inheritance
from keras import initializations
from keras import backend as K
from keras.models import Sequential


class CustomLayer(Layer):
    def __init__(self, hidden_dim, **kwargs):
        """
        Hidden_dim must be defined by the user and so it should be in every constructor
        Goal of this function:
        ----------------------
        * save parameters as self parameters
        * call layers.Layer constructor
        """
        self.hidden_dim = hidden_dim
        self.init = initializations.glorot_uniform

        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Parameters declaration and initialization
        :param input_shape: It is usually of the form (batch_size, sequence_length, input dimension)
        :return:
        """
        self.embedded_dim = input_shape[1]
        self.W_n = self.init((self.embedded_dim, self.hidden_dim))
        self.W_p = self.init((self.embedded_dim, self.hidden_dim))

        self.b = self.init((self.embedded_dim,))

    def call(self, x, mask=None):
        """w as explained before, our inout is of dimension (batch_size, Embedding_dimension, input_sequence_length
		# Now since this model predicts existence of label between a pair of node, input sequence length is just 2
		# We present input to the overall model as pairs of pairs with corresponding labels (node1_index, node2_index, label)
		# We need to convert node_index to a continuous vector (embedding), Embedding layer in Keras handles that. Will come to that part later
		"""

        h_star = x[:, :, 0] * x[:, :, 1]
        h_plus = K.abs(x[:, :, 0] - x[:, :, 1])

        z = K.tanh(K.dot(h_star, self.W_n) + K.dot(h_plus, self.W_p))
        return z

    def get_output_shape_for(self, input_shape):
        """
        If we are changing the dimension, we need to tell it to Keras
        :param input_shape:
        :return:
        """
        return (input_shape[0], self.hidden_dim)


def build_model(embed_dim=10,
                input_size=10,  # Size of the vocabulary
                length_seq=2  # 2 because we have 2 nodes
                ):
    model = Sequential()
    model.add(Embedding(input_dim=input_size,
                        output_dim=embed_dim,
                        input_length=length_seq)) # make a vector of form (batch_size, input_length, embedding_dim
    model.add(Permute([2, 1]))

    model.add(CustomLayer(32))
    model.add(Dense(10))
    model.add(Dense(1, activation='softmax'))
    return model


if __name__ == '__main__':
    build_model()
