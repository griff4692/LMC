import itertools
from multiprocessing import Pool
import os
import sys
from time import time

import argparse
import numpy as np
import pandas as pd

home_dir = os.path.expanduser('~/LMC/')
shared_data = os.path.join(home_dir, 'shared_data')
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
from extract_context_utils import ContextExtractor, ContextType

LFS = pd.read_csv(os.path.join(shared_data, 'casi/labeled_sf_lf_map.csv'))['target_label'].unique().tolist()
CONTEXT_EXTRACTOR = ContextExtractor()


def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)


def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


def add_counts():
    context_fn = 'data/mimic_contexts.csv'
    contexts = pd.read_csv(context_fn)
    lf_counts = dict(contexts['lf'].value_counts())

    lfs = pd.read_csv(os.path.join(shared_data, 'casi', 'labeled_sf_lf_map.csv'))
    counts = []
    for lf in lfs['target_label'].tolist():
        count = 0 if not lf in lf_counts else lf_counts[lf]
        counts.append(count)
    lfs['count'] = counts
    out_fn = 'data/lfs_w_counts.csv'
    lfs.to_csv(out_fn, index=False)


def render_stats():
    in_fn = 'data/lfs_w_counts.csv'
    df = pd.read_csv(in_fn)
    count_freqs = dict(df['count'].value_counts())
    N = df.shape[0]
    print('N={}'.format(N))
    print('0 count={}'.format(count_freqs[0]))
    less_five = 0
    for i in range(5):
        less_five += count_freqs[i]
    print('LFs with less than 5 contexts={}'.format(less_five))

    sfs = df['sf'].unique().tolist()
    viable_sfs = 0
    for sf in sfs:
        subset_df = df[df['sf'] == sf]
        counts = subset_df['count'].tolist()
        if min(counts) > 1:
            viable_sfs += 1
    print('SFs with at least 2 contexts for every LF={}'.format(viable_sfs))


def get_lf_contexts(row, window=15):
    doc_id, doc_category, doc_string = row['ROW_ID'], row['CATEGORY'], row['TEXT']
    contexts, doc_ids, forms, actual_lfs = [], [], [], []
    config = {'type': ContextType.WORD, 'size': window}
    for lf in LFS:
        result = CONTEXT_EXTRACTOR.get_contexts_for_long_form(
            lf, doc_string, config, allow_inflections=False, ignore_case=True)
        for actual_lf, c in result:
            forms.append(lf)
            contexts.append(c)
            actual_lfs.append(actual_lf)
    doc_ids += [doc_id] * len(contexts)
    doc_categories = [doc_category] * len(contexts)
    return list(zip(forms, actual_lfs, doc_ids, doc_categories, contexts))


def extract_mimic_chunk_contexts(input):
    chunk_idx, mimic_df = input
    mimic_data = mimic_df.to_dict('records')
    contexts = map(get_lf_contexts, mimic_data)
    contexts_flat = list(itertools.chain(*contexts))
    df = pd.DataFrame(contexts_flat, columns=['lf', 'lf_match', 'doc_id', 'category', 'context'])
    df.to_csv('data/mimic_context_batches/{}.csv'.format(chunk_idx), index=False)
    return True


def extract_mimic_contexts(chunksize):
    tmp_batch_dir = 'data/mimic_context_batches/'
    if not os.path.exists(tmp_batch_dir):
        print('Creating dir={}'.format(tmp_batch_dir))
        os.mkdir(tmp_batch_dir)

    in_fp = '/nlp/corpora/mimic/mimic_iii/NOTEEVENTS.csv'
    if not os.path.exists(in_fp):
        in_fp = '../data/mimic/NOTEEVENTS.csv'
    print('Loading MIMIC from {}'.format(in_fp))
    mimic_df = pd.read_csv(in_fp)

    target_size = int(mimic_df.shape[0] / float(chunksize - 1))
    mimic_chunk_dfs = split(mimic_df, target_size)
    print('Split into {} chunks'.format(len(mimic_chunk_dfs)))

    start_time = time()
    p = Pool()
    statuses = p.map(extract_mimic_chunk_contexts, enumerate(mimic_chunk_dfs))
    p.close()
    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))
    print(statuses)


def collect_contexts():
    tmp_batch_dir = 'data/mimic_context_batches/'
    fns = os.listdir(tmp_batch_dir)
    df_arr = []
    for fn in fns:
        full_fn = os.path.join(tmp_batch_dir, fn)
        sub_df = pd.read_csv(full_fn)
        print('Adding {} rows from {}'.format(sub_df.shape[0], full_fn))
        df_arr.append(sub_df)
    df = pd.concat(df_arr)
    out_fn = 'data/mimic_contexts.csv'
    print('Saving {} rows to {}'.format(df.shape[0], out_fn))
    df.to_csv(out_fn, index=False)


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC-III Note Acronym Expansion Context Extraction.')
    arguments.add_argument('-add_counts', default=False, action='store_true')
    arguments.add_argument('--chunksize', default=16, type=int)
    arguments.add_argument('-collect', default=False, action='store_true')
    arguments.add_argument('-render_stats', default=False, action='store_true')

    args = arguments.parse_args()

    if args.collect:
        collect_contexts()
    elif args.add_counts:
        add_counts()
    elif args.render_stats:
        render_stats()
    else:
        extract_mimic_contexts(args.chunksize)
