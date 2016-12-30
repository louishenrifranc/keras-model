"""
Create an Embedding Layer and then
a recurrent neural network to recognize if a sentence is talking about a male or a female
"""
from recurrentshop import RecurrentContainer
from keras.layers import Input, Embedding, SimpleRNN
from keras import models
import itertools
from keras import backend as K
import numpy as np

# PREPROCESSING DATA
(X, y) = ("""He is a very tall boy
I like this man so much
What a lovely and cute boy
He will surely succeed in school
He is a talented man indeed
She is very smart and tall
She is a very talented woman
Also, I don't understand her humor
That girl is amazing and young
I think she like that man""".strip().split('\n'), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
assert (len(X) == len(y))
# Clean sentences
lemma = lambda x: x.strip().lower().split(' ')
sentences = [lemma(x) for x in X]

# Create dictionaries
words = set(itertools.chain(*sentences))  # Iterate over each word in each sentences
word2idx = dict((word, index) for index, word in enumerate(words))
idx2word = list(word2idx)

to_idx = lambda x: [word2idx[word] for word in x]
sentences_idx = [to_idx(sentence) for sentence in sentences]
# convert the sentences a numpy array
sentences_array = np.asarray(sentences_idx)

# BUILDING THE MODEL
# Max len of a sentence
sentences_max_len = max(len(sentence) for sentence in sentences)
# Vocabulary size
vocab_size = len(words)
# Embedding size
embedding_size = 3

input = Input(shape=(sentences_max_len,), dtype='int32')
embedding_layer = Embedding(vocab_size, embedding_size)(
    input)  # Create a vector of dim (batch_size, seq_len, embedding_dim)
rnn = SimpleRNN(1)(embedding_layer)

model = models.Model(input, rnn)

model.compile('adadelta', 'binary_crossentropy')
model.fit(sentences_array, y, nb_epoch=100, verbose=2, shuffle=True, callbacks=[])
embeddings = K.get_value(model.layers[1].W)

# PLOT EMBEDDINGS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.subplot(111, projection='3d')
for i in range(vocab_size):  # plot each point + it's index as text above
    ax.scatter(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], color='b')
    ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], '%s' % (str(idx2word[i])), size=20, zorder=1,
            color='k')
plt.show()

# PRINT EMBEDDINGS
for i in range(vocab_size):
    print('{}: {}'.format(idx2word[i], embeddings[i]))
