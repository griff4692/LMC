from collections import Counter, defaultdict
import re
import string
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from mimic_tokenize import clean_text, STOPWORDS, tokenize_str
from model_utils import tensor_to_np


def get_known_ids(vocab, tokens):
    return list(filter(lambda id: id > -1, list(map(lambda tok: vocab.get_id(tok), tokens))))


def point_similarity(model, vocab, tokens_a, tokens_b):
    ids_a = get_known_ids(vocab, tokens_a)
    ids_b = get_known_ids(vocab, tokens_b)

    if len(ids_a) == 0 or len(ids_b) == 0:
        return 0.0

    embeddings = tensor_to_np(model.embeddings_mu.weight)

    rep_a = embeddings[ids_a, :].mean(0)
    rep_b = embeddings[ids_b, :].mean(0)
    sim = 1.0 - cosine(rep_a, rep_b)
    return sim


TOKEN_BLACKLIST = set(string.punctuation).union(STOPWORDS).union(set(['digitparsed']))
# UMLS concepts annoyingly include these terms quite frequently
UMLS_BLACKLIST = TOKEN_BLACKLIST.union(set(['unidentified', 'otherwise', 'specified', 'nos', 'procedure']))


def lf_tokenizer(str, vocab, chunker=None, combine_phrases=False, max_lf_len=5):
    tokens_sep = str.split(';')
    token_bag = set()
    token_counts = defaultdict(int)
    for token_str in tokens_sep:
        tokens = tokenize_str(token_str, combine_phrases=combine_phrases, chunker=chunker)
        for t in tokens:
            token_bag.add(t)
    token_counts[t] += 1
    token_bag = list(token_bag)
    tokens = list(filter(lambda x: x not in UMLS_BLACKLIST and vocab.get_id(x) > -1, token_bag))
    if len(tokens) == 0:
        assert len(token_bag) > 0
        tokens = token_bag
    if len(tokens) > max_lf_len:
        available_token_counts = {}
        for t in tokens:
            available_token_counts[t] = token_counts[t]
        truncated_token_counts = Counter(available_token_counts).most_common(max_lf_len)
        tokens = list(map(lambda x: x[0], truncated_token_counts))
    return tokens


def eval_tokenize(str, unique_only=False, chunker=None, combine_phrases=False):
    str = re.sub(r'_%#(\S+)#%_', r'\1', str)
    str = clean_text(str)
    tokens = tokenize_str(str, combine_phrases=combine_phrases, chunker=chunker)
    tokens = list(filter(lambda t: t not in TOKEN_BLACKLIST, tokens))

    if unique_only:
        tokens = list(set(tokens))
    return tokens


def preprocess_minnesota_dataset(window=5, chunker=None, combine_phrases=False):
    in_fp = 'eval_data/minnesota/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt'
    phrase_str = '_phrase' if combine_phrases else ''
    out_fp = 'eval_data/minnesota/preprocessed_dataset_window_{}{}.csv'.format(window, phrase_str)
    # cols = ['sf', 'target_lf', 'sf_rep', 'start_idx', 'end_idx', 'section', 'context']
    df = pd.read_csv(in_fp, sep='|')
    df.dropna(subset=['sf', 'target_lf', 'context'], inplace=True)

    # Tokenize
    sf_occurrences = []  # When multiple of SF in context, keep track of which one the label is for
    tokenized_contexts = []

    valid_rows = []
    print('Filtering out invalid rows...')
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        sf_idxs = [m.start() for m in re.finditer(r'\b({})\b'.format(row['sf']), row['context'])]
        target_start_idx = int(row['start_idx'])

        valid = ' ' in row['context'][target_start_idx + len(row['sf']): target_start_idx + len(row['sf']) + 2]

        sf_occurrence_ct = np.where(np.array(sf_idxs) == target_start_idx)[0]
        if not valid or len(sf_occurrence_ct) == 0:
            # print('SF not present in context for row={}'.format(row_idx))
            valid_rows.append(False)
            tokenized_contexts.append(None)
            sf_occurrences.append(None)
        else:
            assert len(sf_occurrence_ct) == 1
            valid_rows.append(True)
            sf_occurrences.append(sf_occurrence_ct[0])
            tokens = eval_tokenize(row['context'], combine_phrases=combine_phrases, chunker=chunker)
            tokenized_contexts.append(' '.join(tokens))

    df['valid'] = valid_rows
    df['tokenized_context'] = tokenized_contexts
    df['sf_occurrences'] = sf_occurrences
    prev_n = df.shape[0]
    df = df[df['valid']]
    n = df.shape[0]
    print('Dropping {} rows because we couldn\'t find SF in the context.'.format(prev_n - n))

    trimmed_contexts = []
    print('Tokenizing and extracting context windows...')
    valid = []
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        sf_label_order = int(row['sf_occurrences'])
        tokens = row['tokenized_context'].split()
        sf_idxs = np.where(np.array(tokens) == row['sf'].lower())[0]

        if len(sf_idxs) == 0 or sf_label_order >= len(sf_idxs):
            valid.append(False)
            trimmed_contexts.append(None)
        else:
            valid.append(True)
            sf_idx = sf_idxs[sf_label_order]
            start_idx = max(0, sf_idx - window)
            end_idx = min(sf_idx + window + 1, len(tokens))
            tc = tokens[start_idx:sf_idx] + tokens[sf_idx + 1: end_idx]
            trimmed_contexts.append(' '.join(tc))

    df['trimmed_tokens'] = trimmed_contexts
    df['valid'] = valid
    prev_n = df.shape[0]
    df = df[df['valid']]
    df.drop(columns='valid', inplace=True)
    print('Issues parsing into tokens for {} out of {} remaining examples'.format(prev_n - df.shape[0], prev_n))
    print('Finished preprocessing dataset.  Now saving it to {}'.format(out_fp))
    df.to_csv(out_fp, index=False)


def w2v_point_similarity(model, t1, t2):
    t1 = t1.split()
    t2 = t2.split()

    t1 = list(filter(lambda x: x in model.wv, t1))
    t2 = list(filter(lambda x: x in model.wv, t2))

    if len(t1) == 0 or len(t2) == 0:
        return 0.0

    rep_1 = np.array([model.wv[t] for t in t1]).mean(0)
    rep_2 = np.array([model.wv[t] for t in t2]).mean(0)
    sim = 1.0 - cosine(rep_1, rep_2)
    return sim
