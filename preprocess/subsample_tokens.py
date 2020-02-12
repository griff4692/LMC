import json
import os
import pickle

import argparse
import numpy as np
from tqdm import tqdm

from tokens_to_ids import tokens_to_ids
from vocab import Vocab
sys.path.insert(0, "D:/git_codes/ClinicalBayesianSkipGram/")


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Subsampling of Already Tokenized Data.')
    arguments.add_argument('--tokenized_fp', default='NOTEEVENTS_tokenized')
    arguments.add_argument('--token_counts_fp', default='NOTEEVENTS_token_counts')

    arguments.add_argument('-combine_phrases', default=False, action='store_true')
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('--min_token_count', default=10, type=int)
    arguments.add_argument('--subsample_param', default=0.001, type=float)
    arguments.add_argument('-split_sentences', default=False, action='store_true')

    args = arguments.parse_args()

    np.random.seed(1992)
    rand_arr = np.random.rand(10000)
    rand_ct = 0

    # Expand home path (~) so that pandas knows where to look
    args.tokenized_fp = os.path.expanduser(args.tokenized_fp)
    args.token_counts_fp = os.path.expanduser(args.token_counts_fp)

    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''
    sentence_str = '_sentence' if args.split_sentences else ''
    tokenized_data_fn = sys.path[0] + '{}{}{}{}.json'.format(args.tokenized_fp, debug_str, phrase_str, sentence_str)
    with open(tokenized_data_fn, 'r') as fd:
        tokenized_data = json.load(fd)
    token_counts_fn = sys.path[0] + '{}{}{}{}.json'.format(args.token_counts_fp, debug_str, phrase_str, sentence_str)
    with open(token_counts_fn, 'r') as fd:
        token_counts = json.load(fd)
    N = float(token_counts['__ALL__'])
    print('Subsampling {} tokens'.format(N))

    # Store subsampled data
    tokenized_subsampled_data = []
    # And vocabulary with word counts
    vocab = Vocab()
    num_docs = len(tokenized_data)
    sections = set()
    categories = set()
    for doc_idx in tqdm(range(num_docs)):
        tokenized_doc_str = tokenized_data[doc_idx]
        subsampled_doc = []
        prev_token = 'dummy'
        doc_tokens = tokenized_doc_str.split()
        for tidx, token in enumerate(doc_tokens):
            if prev_token == token:
                continue
            wc = token_counts[token]
            is_section_header = 'header=' in token
            is_doc_header = 'document=' in token
            if is_section_header:
                if not tidx + 1 == len(doc_tokens) and not 'header=' in doc_tokens[tidx + 1]:
                    subsampled_doc.append(token)
                    sections.add(token)
            elif is_doc_header:
                subsampled_doc.append(token)
                categories.add(token)
            else:
                if wc < args.min_token_count:
                    continue
                frac = wc / N
                keep_prob = min((np.sqrt(frac / args.subsample_param) + 1) * (args.subsample_param / frac), 1.0)

                should_keep = rand_arr[rand_ct] < keep_prob

                rand_ct += 1
                if rand_ct == len(rand_arr):
                    rand_ct = 0
                    rand_arr = np.random.rand(10000)
                if should_keep:
                    subsampled_doc.append(token)
                    vocab.add_token(token, token_support=1)
                else:
                    continue
            prev_token = token
        tokenized_subsampled_data.append(' '.join(subsampled_doc))

    print('Reduced tokens from {} to {}'.format(int(N), sum(vocab.support)))
    vocab.section_start_vocab_id = vocab.size()
    print('Adding {} section headers'.format(len(sections)))
    vocab.add_tokens(sections, token_support=0)
    vocab.category_start_vocab_id = vocab.size()
    print('Adding {} document categories'.format(len(categories)))
    vocab.add_tokens(categories, token_support=0)

    subsampled_out_fn = sys.path[0] + '{}_subsampled{}{}{}.json'.format(args.tokenized_fp, debug_str, phrase_str, sentence_str)
    print('Saving subsampled tokens to {}'.format(subsampled_out_fn))
    with open(subsampled_out_fn, 'w') as fd:
        json.dump(tokenized_subsampled_data, fd)
    vocab_out_fn = sys.path[0] + 'vocab{}{}{}.pk'.format(debug_str, phrase_str, sentence_str)
    print('Saving vocabulary of size {} to {}'.format(vocab.size(), vocab_out_fn))
    with open(vocab_out_fn, 'wb') as fd:
        pickle.dump(vocab, fd)

    print('Converting to id matrix...')
    tokens_to_ids(args, token_infile=subsampled_out_fn,sys_path = sys.path[0])
