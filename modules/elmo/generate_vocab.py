import os
import pickle
import sys

import argparse
import numpy as np

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Vocabulary in AllenNLP Format')

    """
    Writes unigram vocabulary where each line is a unique word type and order is by inverse frequency.
    Does not include metadata tokens as these are not modeled under ELMo.
    """

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)

    args = parser.parse_args()
    # Load Data
    debug_str = '_mini' if args.debug else ''
    vocab_infile = os.path.join(home_dir, 'preprocess/data/vocab{}.pk'.format(debug_str))
    print('Loading vocabulary from {}...'.format(vocab_infile))
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    split_idx = min(vocab.section_start_vocab_id, vocab.category_start_vocab_id)
    print('Loaded vocabulary of size={}...'.format(split_idx))
    vocab_order = ['</S>', '<S>', '@@UNKNOWN@@']
    tokens = vocab.i2w[1:split_idx]
    supports = vocab.support[1:split_idx]

    token_order = np.argsort(-np.array(supports))
    tokens_ordered = list(np.array(tokens)[token_order])
    vocab_order += tokens_ordered
    out_fn = 'data/vocab/tokens.txt'
    print('Saving {} tokens to {}'.format(len(vocab_order), out_fn))
    with open(out_fn, 'w') as fd:
        fd.write('\n'.join(vocab_order))
