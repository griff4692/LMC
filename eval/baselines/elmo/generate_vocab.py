import pickle

import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-combine_phrases', default=False, action='store_true')

    args = parser.parse_args()
    # Load Data
    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''
    vocab_infile = '../../../preprocess/data/vocab{}{}.pk'.format(debug_str, phrase_str)
    print('Loading vocabulary from {}...'.format(vocab_infile))
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    print('Loaded vocabulary of size={}...'.format(vocab.separator_start_vocab_id))
    vocab_order = ['</S>', '<S>', '@@UNKNOWN@@']

    tokens = vocab.i2w[1:vocab.separator_start_vocab_id]
    supports = vocab.support[1:vocab.separator_start_vocab_id]

    token_order = np.argsort(-np.array(supports))
    tokens_ordered = list(np.array(tokens)[token_order])
    vocab_order += tokens_ordered
    with open('data/vocab/tokens.txt', 'w') as fd:
        fd.write('\n'.join(vocab_order))
