from collections import defaultdict
from multiprocessing import Pool
import os
import sys
from time import time

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def add_counts(dataset):
    context_fn = 'data/{}_contexts.csv'.format(dataset)
    contexts = pd.read_csv(context_fn)
    lf_counts = dict(contexts['lf'].value_counts())

    lfs = pd.read_csv(os.path.join(shared_data, 'casi/labeled_sf_lf_map.csv'))
    counts = []
    for lf in lfs['target_label'].tolist():
        count = 0 if not lf in lf_counts else lf_counts[lf]
        counts.append(count)
    lfs['count'] = counts
    out_fn = 'data/{}_lfs_w_counts.csv'.format(dataset)
    lfs.to_csv(out_fn, index=False)


def collect_contexts(dataset):
    tmp_batch_dir = 'data/{}_context_batches/'.format(dataset)
    fns = os.listdir(tmp_batch_dir)
    df_arr = []
    for fn in fns:
        full_fn = os.path.join(tmp_batch_dir, fn)
        sub_df = pd.read_csv(full_fn)
        print('Adding {} rows from {}'.format(sub_df.shape[0], full_fn))
        df_arr.append(sub_df)
    df = pd.concat(df_arr)
    out_fn = 'data/{}_contexts.csv'.format(dataset)
    print('Saving {} rows to {}'.format(df.shape[0], out_fn))
    df.to_csv(out_fn, index=False)


def render_stats(dataset):
    in_fn = 'data/{}_lfs_w_counts.csv'.format(dataset)
    df = pd.read_csv(in_fn)
    count_freqs = dict(df['count'].value_counts())
    N = df.shape[0]
    print('N={}'.format(N))
    print('0 count={}'.format(count_freqs[0]))
    less_five = 0
    for i in range(0, 5):
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


def get_lf_contexts(row, window=20):
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


def extract_contexts(dataset, chunk, chunksize, debug):
    debug_str = '_mini' if debug else ''
    if dataset == 'mimic':
        chunk_data = get_mimic_chunk(chunk, chunksize, debug)
    else:
        chunk_data = get_columbia_chunk(chunk, chunksize)
    print('Loaded {} documents at chunk {}'.format(len(chunk_data), chunk))
    start_time = time()
    p = Pool()
    contexts = p.map(get_lf_contexts, chunk_data)
    p.close()
    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))

    contexts_flat = []
    for x in contexts:
        for y in x:
            contexts_flat.append(y)

    df = pd.DataFrame(contexts_flat, columns=['lf', 'lf_match', 'doc_id', 'category', 'context'])
    df.to_csv('data/{}_context_batches/{}{}.csv'.format(dataset, chunk, debug_str), index=False)


def get_mimic_chunk(chunk, chunksize, debug):
    tmp_batch_dir = 'data/mimic_context_batches/'
    if not os.path.exists(tmp_batch_dir):
        print('Creating dir={}'.format(tmp_batch_dir))
        os.mkdir(tmp_batch_dir)

    debug_str = '_mini' if debug else ''
    in_fp = os.path.join(home_dir, 'preprocess/data/mimic/NOTEEVENTS{}.csv'.format(debug_str))
    print('Loading MIMIC from {}'.format(in_fp))
    mimic_df = pd.read_csv(in_fp)

    if chunksize > 1:
        target_size = int(mimic_df.shape[0] / float(chunksize))
        mimic_df = split(mimic_df, target_size)[chunk]
    return mimic_df.to_dict('records')


def read_columbia_dataset():
    columbia_data = defaultdict(list)
    in_fp = '/nlp/projects/BERT_corpus_250M/corpusFiles/train/corpus.txt'
    print('Loading Columbia data from {}'.format(in_fp))
    curr_doc = ''
    with open(in_fp, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            if len(line) == 0 and len(curr_doc) > 0:
                columbia_data['ROW_ID'].append(i)
                columbia_data['TEXT'].append(curr_doc.strip())
                columbia_data['CATEGORY'].append('Columbia Notes')
                curr_doc = ''
            else:
                curr_doc += line + ' '
            if (i + 1) % 1000000 == 0:
                print('Processed {} lines'.format(i + 1))
    if len(curr_doc) > 0:
        columbia_data['ROW_ID'].append(i)
        columbia_data['TEXT'].append(curr_doc.strip())
        columbia_data['CATEGORY'].append('Columbia Notes')
    columbia_df = pd.DataFrame(columbia_data)
    return columbia_df


def get_columbia_chunk(chunk, chunksize):
    tmp_batch_dir = 'data/columbia_context_batches/'
    if not os.path.exists(tmp_batch_dir):
        print('Creating dir={}'.format(tmp_batch_dir))
        os.mkdir(tmp_batch_dir)

    columbia_df = read_columbia_dataset()
    if chunksize > 1:
        target_size = int(columbia_df.shape[0] / float(chunksize))
        columbia_chunk_df = split(columbia_df, target_size)[chunk]
    else:
        columbia_chunk_df = columbia_df
    return columbia_chunk_df.to_dict('records')


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('Clinical Note Acronym Expansion Context Extraction.')
    arguments.add_argument('-add_counts', default=False, action='store_true')
    arguments.add_argument('--chunk', default=0, type=int)
    arguments.add_argument('--chunksize', default=10, type=int)
    arguments.add_argument('-collect', default=False, action='store_true')
    arguments.add_argument('--dataset', default='mimic')
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('-render_stats', default=False, action='store_true')

    args = arguments.parse_args()

    if args.collect:
        collect_contexts(args.dataset)
    elif args.add_counts:
        add_counts(args.dataset)
    elif args.render_stats:
        render_stats(args.dataset)
    else:
        extract_contexts(args.dataset, args.chunk, args.chunksize, args.debug)
