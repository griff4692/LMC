"""
Embeddings borrowed from https://github.com/ncbi-nlp/BioSentVec/
"""

import os

import argparse
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle
from tqdm import tqdm

from vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MIMIC (v3) Collect Pretrained Embeddings.')
    parser.add_argument('--biowordvec_fp', default='~/BioWordVec_PubMed_MIMICIII_d200.vec.bin')
    parser.add_argument('-debug', action='store_true', default=False)

    args = parser.parse_args()

    # Expand home path (~) so that pandas knows where to look
    debug_str = '_mini' if args.debug else ''
    args.biowordvec_fp = os.path.expanduser(args.biowordvec_fp)
    vocab_infile = '../preprocess/data/vocab{}.pk'.format(debug_str)
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)

    print('Loading pre-trained vectors...')
    vectors = KeyedVectors.load_word2vec_format(args.biowordvec_fp, binary=True)

    DIM = vectors.vector_size
    embeddings = np.zeros([vocab.size(), DIM])
    oov = set()
    print('Generating embedding matrix for vocab of size={}'.format(vocab.size()))
    for idx in tqdm(range(1, vocab.size())):
        token = vocab.get_token(idx)
        if token in vectors:
            embeddings[idx, :] = vectors[token]
        else:
            embeddings[idx, :] = np.random.normal(loc=0.0, scale=1.0, size=(DIM, ))
            oov.add(token)
    print('Randomly initialized the following tokens...')
    print(', '.join(list(oov)))
    np.save('./data/embeddings{}.npy'.format(debug_str), embeddings)
