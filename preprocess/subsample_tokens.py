import json
import os

import argparse
import numpy as np

from preprocess.vocab import Vocab


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Subsampling of Already Tokenized Data.')
    arguments.add_argument('--tokenized_fp', default='~/Desktop/mimic/NOTEEVENTS_tokenized')
    arguments.add_argument('--token_counts_fp', default='~/Desktop/mimic/NOTEEVENTS_token_counts')

    arguments.add_argument('--min_token_count', default=5, type=int)
    arguments.add_argument('--subsample_param', default=0.001, type=float)

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    args.tokenized_fp = os.path.expanduser(args.tokenized_fp + '.json')
    args.token_counts_fp = os.path.expanduser(args.token_counts_fp + '.json')

    tokenized_data = json.load(open(args.tokenized_fp, 'r'))
    token_counts = json.load(open(args.token_counts_fp, 'r'))
    N = float(token_counts['__ALL__'])

    tokenized_subsampled_data = []

    vocab = Vocab()
    num_original_tokens, num_subsampled_tokens = 0, 0
    for category, tokenized_doc_str in tokenized_data:
        subsampled_doc = []
        for token in tokenized_doc_str.split():
            num_original_tokens += 1
            wc = token_counts[token]
            frac = wc / N
            too_sparse = wc < args.min_token_count
            keep_prob = min((np.sqrt(frac / args.subsample_param) + 1) * (args.subsample_param / frac), 1.0)
            too_frequent = np.random.random() > keep_prob
            if not too_sparse and not too_frequent:
                subsampled_doc.append(token)
                num_subsampled_tokens += 1
                vocab.add_token(token)
        tokenized_subsampled_data.append((category, ' '.join(subsampled_doc)))
    print('Reduced tokens from {} to {}'.format(num_original_tokens, num_subsampled_tokens))
    print('Saving vocabulary of size {}'.format(vocab.size()))
    json.dump(tokenized_subsampled_data, open(args.tokenized_fp + 'subsampled.json', 'w'))
    np.save('./data/vocab.npy', vocab)
