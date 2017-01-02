from gensim.models import Word2Vec
from keras.layers import Input, Embedding
import numpy as np
import json


def create_embeddings(data_dir, embedding_path, vocab_path, **kwargs):
    class SentencesGenerator(object):
        def __init__(self, filename):
            self.filename = filename

        def __iter__(self):
            with open(self.filename) as f:
                for line in f:
                    yield line

    sentences = SentencesGenerator('messenger.htm')
    model = Word2Vec(sentences, **kwargs)

    weights = model.syn0
    np.save(open(embedding_path, 'wb'), weights)

    vocab = dict([(k, v.index for k, v in model.vocab.items())])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))


def load_embeddings(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.load(f.read())

    word2idx = data
    idx2word = list(word2idx)
    return word2idx, idx2word


def word2vec_Embedding(embedding_path):
    weights = np.load(open(embedding_path, 'r'))
    embedding_layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=weights)
    return embedding_layer


