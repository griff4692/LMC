import pickle
import sys

from gensim.models import Word2Vec
import numpy as np


sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from vocab import Vocab


if __name__ == '__main__':
    ids_infile = '../../../preprocess/data/ids.npy'
    vocab_infile = '../../../preprocess/data/vocab.pk'

    print('Loading vocabulary...')
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    token_vocab_size = vocab.size()
    print('Loaded vocabulary of size={}...'.format(token_vocab_size))

    docs = []
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)

    curr_doc = []
    for ct, id in enumerate(ids):
        if id == 0:
            docs.append(curr_doc)
            curr_doc = []
        elif id < 0:
            curr_doc.append(vocab.get_token(-id))
        else:
            curr_doc.append(vocab.get_token(id))
        if (ct + 1) % 1000000 == 0:
            print('Processed {} tokens'.format(ct + 1))

    if len(curr_doc) > 0:
        docs.append(curr_doc)

    neg_samples = [1, 5, 20]
    for ns in neg_samples:
        print('Training with negative samples={}'.format(ns))
        model = Word2Vec(docs, size=100, window=5, min_count=1, iter=10, workers=10, sample=1, negative=ns)
        model.save("weights/word2vec_ns_{}.model".format(ns))
