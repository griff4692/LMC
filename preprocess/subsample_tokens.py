import json
import os
import pickle

import argparse
import numpy as np
from tqdm import tqdm

from tokens_to_ids import tokens_to_ids
from vocab import Vocab


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Subsampling of Already Tokenized Data.')
    arguments.add_argument('--tokenized_fp', default='data/mimic/NOTEEVENTS_tokenized')
    arguments.add_argument('--token_counts_fp', default='data/mimic/NOTEEVENTS_token_counts')

    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('--min_token_count', default=5, type=int)
    arguments.add_argument('--subsample_param', default=0.001, type=float)

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    args.tokenized_fp = os.path.expanduser(args.tokenized_fp)
    args.token_counts_fp = os.path.expanduser(args.token_counts_fp)

    debug_str = '_mini' if args.debug else ''
    tokenized_data_fn = '{}{}.json'.format(args.tokenized_fp, debug_str)
    with open(tokenized_data_fn, 'r') as fd:
        tokenized_data = json.load(fd)
    token_counts_fn = '{}{}.json'.format(args.token_counts_fp, debug_str)
    with open(token_counts_fn, 'r') as fd:
        token_counts = json.load(fd)
    N = float(token_counts['__ALL__'])

    # Store subsampled data
    tokenized_subsampled_data = []
    # And vocabulary with word counts
    vocab = Vocab()
    num_docs = len(tokenized_data)
    for doc_idx in tqdm(range(num_docs)):
        category, tokenized_doc_str = tokenized_data[doc_idx]
        subsampled_doc = []
        for token in tokenized_doc_str.split():
            wc = token_counts[token]
            too_sparse = wc <= args.min_token_count
            if too_sparse:
                continue
            frac = wc / N
            keep_prob = min((np.sqrt(frac / args.subsample_param) + 1) * (args.subsample_param / frac), 1.0)
            should_keep = np.random.binomial(1, keep_prob) == 1
            if should_keep:
                subsampled_doc.append(token)
                vocab.add_token(token, token_support=1)
        tokenized_subsampled_data.append((category, ' '.join(subsampled_doc)))

    print('Reduced tokens from {} to {}'.format(int(N), sum(vocab.support)))
    print('Saving vocabulary of size {}'.format(vocab.size()))
    subsampled_out_fn = args.tokenized_fp + ('_subsampled_mini.json' if args.debug else '_subsampled.json')
    with open(subsampled_out_fn, 'w') as fd:
        json.dump(tokenized_subsampled_data, fd)
    vocab_out_fn = './data/vocab_mini.pk' if args.debug else './data/vocab.pk'
    with open(vocab_out_fn, 'wb') as fd:
        pickle.dump(vocab, fd)

    print('Converting to id matrix...')
    tokens_to_ids(args, token_infile=subsampled_out_fn)
