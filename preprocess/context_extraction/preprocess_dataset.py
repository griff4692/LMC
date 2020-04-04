import json
import os
import re
import string
import sys

import numpy as np
import pandas as pd

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
from mimic_tokenize import create_section_token, is_actual_header, preprocess_mimic
from compute_sections import HEADER_SEARCH_REGEX
from canonical_sections import get_canonical_section


WINDOW = 10  # Context token window (10 tokens on either side of acronym / SF)


def get_header_from_full_context(mimic_df, context, lf_match, doc_id, category, sub_header=False):
    """
    :param mimic_df: NOTEEVENTS.csv dataframe
    :param context: local context surrounding expansion
    :param lf_match: target LF (center of context)
    :param doc_id: ROW_ID in which context was found
    :return: first header preceding context, if any

    Sometimes the window of text returned contains no section headers, which means we must go back and search through
    whole document to find appropriate section header.
    """
    full_context = mimic_df[mimic_df['ROW_ID'] == doc_id]['TEXT'].tolist()[0]
    context_repl = context.replace('TARGETWORD', lf_match)
    try:
        pre_idx = full_context.index(context_repl)
    except:
        try:
            pre_idx = full_context.index(context.split('TARGETWORD')[0].strip()[:-3])
        except:
            print('here')
            if sub_header:
                return ''
            else:
                return create_section_token('{}->START'.format(category))
    relevant_context = full_context[:pre_idx]
    sectioned_text = list(filter(
        lambda x: len(x.strip()) > 0, re.split(HEADER_SEARCH_REGEX, relevant_context, flags=re.M)))
    is_header_arr = list(map(lambda x: re.match(HEADER_SEARCH_REGEX, x, re.M) is not None, sectioned_text))
    headers = []
    sub_headers = []
    for tok_idx, is_header in enumerate(is_header_arr):
        tok = sectioned_text[tok_idx]
        if is_header and is_actual_header(tok):
            header_stripped = tok.strip().strip(':').upper()
            canonical_section = get_canonical_section(header_stripped, category)
            is_canonical = '->' in canonical_section
            header = create_section_token(canonical_section, is_canonical)
            if is_canonical:
                headers.append(header)
            else:
                sub_headers.append(header)

    if sub_header:
        if len(sub_headers) == 0:
            return ''
        return sub_headers[-1]
    else:
        if len(headers) == 0:
            return create_section_token('{}->START'.format(category))
        return headers[-1]


def preprocess_mimic_rs():
    """
    :return: None

    Filters out invalid examples, tokenizes, and returns preprocessed dataset ready for evaluation.
    """
    casi_dir = os.path.join(home_dir, 'shared_data', 'casi')
    with open(os.path.join(casi_dir, 'sf_lf_map.json'), 'r') as fd:
        sf_lf_map = json.load(fd)

    remote_path = '/nlp/corpora/mimic/mimic_iii/NOTEEVENTS.csv'
    if os.path.exists(remote_path):
        mimic_df = pd.read_csv(remote_path)
    else:
        mimic_df = pd.read_csv(os.path.join(home_dir, 'preprocess/data/mimic/NOTEEVENTS.csv'))
    df = pd.read_csv('data/mimic_rs_dataset.csv')
    N = df.shape[0]
    tokenized_contexts = []
    metadata = []
    sub_sections = []
    trimmed_contexts = []
    trimmed_sentences = []
    lf_tokenized_out = set()

    print('Tokenizing and extracting context windows...')
    is_valid = []
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        context = row['context']
        category = row['category']
        tokens_orig = preprocess_mimic((category, context)).split(' ')[1:]
        tokens = []
        for token in tokens_orig:
            for t in re.split(r'(targetword)', token):
                if not t == '<sep>':
                    t = t.strip(string.punctuation)
                if len(t) > 0:
                    tokens.append(t)
        is_header = list(map(lambda x: 'header=' in x, tokens))
        header_locs = np.where(np.array(is_header))[0]
        sep_locs = np.where(np.array(tokens) == '<sep>')[0]
        sf_idx = np.where(np.array(tokens) == 'targetword')[0]

        if not len(sf_idx) == 1:
            is_valid.append(False)
            tokenized_contexts.append(None)
            metadata.append(None)
            sub_sections.append(None)
            trimmed_contexts.append(None)
            trimmed_sentences.append(None)
            # print('LF={} was tokenized out of context'.format(row['lf']))
            lf_tokenized_out.add(row['lf'])
        else:
            sf_idx = sf_idx[0]
            if len(header_locs) > 0:
                header_rel = header_locs - sf_idx
                header_rel_pos_mask = np.where(header_rel > 0)[0]
                header_rel_neg_mask = np.where(header_rel < 0)[0]

                header_rel_left = header_rel.copy()
                header_rel_left[header_rel_pos_mask] = -100

                header_rel_right = header_rel.copy()
                header_rel_right[header_rel_neg_mask] = 100

                left_header_window = header_locs[header_rel_left.argmax()]
                right_header_window = header_locs[header_rel_right.argmin()]

                if left_header_window > sf_idx:
                    left_header_window = None
                if right_header_window < sf_idx:
                    right_header_window = None
            else:
                left_header_window = None
                right_header_window = None

            if len(sep_locs) > 0:
                sep_rel = sep_locs - sf_idx
                sep_rel_pos_mask = np.where(sep_rel > 0)[0]
                sep_rel_neg_mask = np.where(sep_rel < 0)[0]

                sep_rel_left = sep_rel.copy()
                sep_rel_left[sep_rel_pos_mask] = -100

                sep_rel_right = sep_rel.copy()
                sep_rel_right[sep_rel_neg_mask] = 100

                left_sep_window = sep_locs[sep_rel_left.argmax()]
                right_sep_window = sep_locs[sep_rel_right.argmin()]

                if left_sep_window > sf_idx:
                    left_sep_window = None
                if right_sep_window < sf_idx:
                    right_sep_window = None
            else:
                left_sep_window = None
                right_sep_window = None

            left_header = tokens[left_header_window] if left_header_window is not None else None
            left_header_boundary = 0 if left_header is None else left_header_window + 1
            left_sep_boundary = 0 if left_sep_window is None else left_sep_window + 1

            if left_header is None:
                left_header = get_header_from_full_context(mimic_df, context, row['lf_match'], row['doc_id'], row['category'])

            if len(left_header) > 100:
                left_header = get_header_from_full_context(mimic_df, context, row['lf_match'], row['doc_id'], row['category'])
            right_header_boundary = len(tokens) if right_header_window is None else right_header_window
            right_sep_boundary = len(tokens) if right_sep_window is None else right_sep_window

            left_window = max(sf_idx - WINDOW, 0, left_header_boundary)
            left_window_sentence = max(left_window, left_sep_boundary)
            right_window = min(sf_idx + WINDOW + 1, len(tokens), right_header_boundary)
            right_window_sentence = min(right_window, right_sep_boundary)
            window_tokens = tokens[left_window:sf_idx] + [row['sf']] + tokens[sf_idx + 1:right_window]
            window_tokens_sentence = tokens[left_window_sentence:sf_idx] + [row['sf']] + tokens[sf_idx + 1:right_window_sentence]

            has_sub = any(['sub=' in t for t in tokens[left_window:sf_idx]])
            sub_header = None
            if not has_sub:
                sub_header = get_header_from_full_context(
                    mimic_df, context, row['lf_match'], row['doc_id'], row['category'], sub_header=True)

            if len(window_tokens) == 0:
                trimmed_contexts.append('<pad>')
                trimmed_sentences.append('<pad>')
            else:
                trimmed_contexts.append(' '.join(window_tokens))
                trimmed_sentences.append(' '.join(window_tokens_sentence))
            metadata.append(left_header)
            sub_sections.append(sub_header)
            tokenized_contexts.append(' '.join(tokens))
            is_valid.append(True)
        if (row_idx + 1) % 1000 == 0:
            print('Processed {} out of {} examples'.format(row_idx + 1, N))

    df['target_lf_idx'] = df['sf'].combine(df['target_lf_sense'], lambda sf, lf: sf_lf_map[sf].index(lf))
    df['row_idx'] = list(range(df.shape[0]))
    df.rename(columns={'lf': 'target_lf'}, inplace=True)
    df['trimmed_tokens'] = trimmed_contexts
    df['tokenized_context'] = tokenized_contexts
    df['tokenized_context_sentence'] = trimmed_sentences
    df['metadata'] = metadata
    df['sub_metadata'] = sub_sections
    df['is_valid'] = is_valid

    df = df[df['is_valid']]
    print('Removing {} invalid rows'.format(N - df.shape[0]))
    print('Tokenize these LFs out of >= 1 examples: = {}'.format(', '.join(list(lf_tokenized_out))))

    prev_N = df.shape[0]
    df.drop_duplicates(subset=['target_lf', 'tokenized_context'], inplace=True)
    N = df.shape[0]
    print('Removing {} duplicate rows. Saving {}'.format(prev_N - N, N))
    df.to_csv('data/mimic_rs_preprocessed.csv', index=False)


if __name__ == '__main__':
    preprocess_mimic_rs()
