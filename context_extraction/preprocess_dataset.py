import json
import sys

import pandas as pd


sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/eval/')
from eval_utils import eval_tokenize


def preprocess_mimic_rs(window=10):
    with open('data/sf_lf_map.json', 'r') as fd:
        used_sf_lf_map = json.load(fd)

    df = pd.read_csv('data/mimic_rs_dataset.csv')
    tokenized_contexts = []
    trimmed_contexts = []
    print('Tokenizing and extracting context windows...')
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        context = row['context']
        context_tokens = eval_tokenize(context)

        tokenized_contexts.append(' '.join(context_tokens))
        for sf_idx, token in enumerate(context_tokens):
            if token == 'targetword':
                break

        left_window = max(sf_idx - window, 0)
        right_window = min(sf_idx + window + 1, len(context_tokens))
        window_tokens = context_tokens[left_window:sf_idx] + context_tokens[sf_idx + 1:right_window]
        trimmed_contexts.append(' '.join(window_tokens))

    df['used_target_lf_idx'] = df['sf'].combine(df['lf_orig'], lambda sf, lf: used_sf_lf_map[sf].index(lf))
    df['row_idx'] = list(range(df.shape[0]))
    df.rename(columns={'lf': 'target_lf'}, inplace=True)
    df['trimmed_tokens'] = trimmed_contexts
    df['tokenized_context'] = tokenized_contexts

    prev_N = df.shape[0]
    df.drop_duplicates(subset=['tokenized_context'], inplace=True)
    N = df.shape[0]
    print('Removing {} duplicate rows. Saving {}'.format(prev_N - N, N))
    df.to_csv('data/mimic_rs_dataset_preprocessed_window_{}.csv'.format(window), index=False)


if __name__ == '__main__':
    preprocess_mimic_rs()
