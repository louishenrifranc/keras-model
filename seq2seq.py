from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, InputLayer
from recurrent

def Seq2Seq(output_dim,
            output_length,
            hidden_dim,
            depth,
            dropout,
            batch_size,
            **kwargs):

