from collections import defaultdict
import json
from multiprocessing import Pool
import os
from time import time

import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from mimic_tokenize import preprocess_mimic


def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)


def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--mimic_fp', default='data/mimic/NOTEEVENTS')
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('--chunk', type=int, required=True)
    arguments.add_argument('--chunksize', type=int, required=True)

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    debug_str = '_mini' if args.debug else ''
    df = pd.read_csv('{}{}.csv'.format(args.mimic_fp, debug_str))
    target_size = int(df.shape[0] / float(args.chunksize))
    df = split(df, target_size)[args.chunk]
    print('Loaded {} rows of data. Tokenizing...'.format(df.shape[0]))
    categories = df['CATEGORY'].tolist()
    start_time = time()
    p = Pool()
    parsed_docs = p.map(preprocess_mimic, df['TEXT'].tolist())
    p.close()

    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))

    token_cts = defaultdict(int)
    for doc_idx, doc in enumerate(parsed_docs):
        for token in doc.split():
            token_cts[token] += 1
            token_cts['__ALL__'] += 1
    debug_str = '_mini' if args.debug else ''
    with open(args.mimic_fp + '_tokenized{}_chunk_{}.json'.format(debug_str, args.chunk), 'w') as fd:
        json.dump(list(zip(categories, parsed_docs)), fd)
    with open(args.mimic_fp + '_token_counts{}_chunk_{}.json'.format(debug_str, args.chunk), 'w') as fd:
        json.dump(token_cts, fd)
