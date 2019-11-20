"""
Embeddings borrowed from https://github.com/ncbi-nlp/BioSentVec/
"""

import os

import argparse
import numpy as np

from gensim.models.keyedvectors import KeyedVectors

if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Collect Pretrained Embeddings.')
    arguments.add_argument('--biowordvec_fp', default='~/Desktop/BioWordVec_PubMed_MIMICIII_d200.vec.bin')

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    args.biowordvec_fp = os.path.expanduser(args.biowordvec_fp)

    vocab = np.load('./data/vocab.npy', allow_pickle=True)
    vectors = KeyedVectors.load_word2vec_format(args.biowordvec_fp, binary=True)

    DIM = 200

    embeddings = np.zeros([vocab.size(), DIM])
    for idx in range(1, vocab.size()):
        token = vocab.get_token(idx)
        embeddings[idx, :] = vectors[token]
    np.save('./data/embeddings.npy', embeddings)
