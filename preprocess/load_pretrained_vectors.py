"""
Embeddings borrowed from https://github.com/ncbi-nlp/BioSentVec/
"""

import os

import argparse
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle

from vocab import Vocab


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Collect Pretrained Embeddings.')
    arguments.add_argument('--biowordvec_fp', default='~/Desktop/BioWordVec_PubMed_MIMICIII_d200.vec.bin')

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    args.biowordvec_fp = os.path.expanduser(args.biowordvec_fp)
    vocab = pickle.load(open('./data/vocab', 'rb'))
    print('Generating embedding matrix for vocab of size={}'.format(vocab.size()))
    vectors = KeyedVectors.load_word2vec_format(args.biowordvec_fp, binary=True)

    DIM = vectors.vector_size
    embeddings = np.zeros([vocab.size(), DIM])
    oov = set()
    for idx in range(1, vocab.size()):
        token = vocab.get_token(idx)
        if token in vectors:
            embeddings[idx, :] = vectors[token]
        else:
            embeddings[idx, :] = np.random.normal(loc=0.0, scale=1.0, size=(DIM, ))
            oov.add(token)
    print('Randomly initialized the following tokens...')
    print(', '.join(list(oov)))
    np.save('./data/embeddings.npy', embeddings)
