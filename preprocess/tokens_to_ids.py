import json
import os
import pickle

import argparse
import numpy as np
from tqdm import tqdm

from vocab import Vocab


def tokens_to_ids(args, token_infile=None):
    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''
    sentence_str = '_sentence' if args.split_sentences else ''

    if token_infile is None:
        token_infile = '{}{}{}{}.json'.format(args.tokenized_fp, debug_str, phrase_str, sentence_str)
    with open(token_infile, 'r') as fd:
        tokens = json.load(fd)

    # Load Vocabulary
    vocab_infile = 'data/vocab{}{}{}.pk'.format(debug_str, phrase_str, sentence_str)
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)

    ids = []
    N = len(tokens)
    for doc_idx in tqdm(range(N)):
        doc_tokens = tokens[doc_idx][1].split()
        doc_ids = vocab.get_ids(doc_tokens)
        doc_ids = [-d if d >= vocab.separator_start_vocab_id else d for d in doc_ids]
        ids += doc_ids
        ids += [0]  # Treating this as document boundary

    ids = ids[:-1]
    print('Saving {} tokens to disc'.format(len(ids)))
    out_fn = 'data/ids{}{}{}.npy'.format(debug_str, phrase_str, sentence_str)
    with open(out_fn, 'wb') as fd:
        np.save(fd, np.array(ids, dtype=int))


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokens to Ids.')

    arguments.add_argument('--tokenized_fp', default='~/Desktop/mimic/NOTEEVENTS_tokenized_subsampled')
    arguments.add_argument('-debug', default=False, action='store_true')

    args = arguments.parse_args()

    args.tokenized_fp = os.path.expanduser(args.tokenized_fp)
    tokens_to_ids(args)
