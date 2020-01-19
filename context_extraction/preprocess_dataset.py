import json
import os
import re
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/eval/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from mimic_tokenize import SECTION_NAMES, clean_text, tokenize_str, create_section_token


def _is_header(sectioned_text, idx):
    if idx == len(sectioned_text):
        return False
    return sectioned_text[idx] in SECTION_NAMES and idx + 1 < len(sectioned_text) and sectioned_text[idx + 1] == ':'


def get_header_from_full_context(mimic_df, context, lf_match, doc_id, header_regexes):
    full_contexts = mimic_df[mimic_df['ROW_ID'] == doc_id]['TEXT'].tolist()
    assert len(full_contexts) == 1
    full_context = re.sub(r'\s+', ' ', full_contexts[0])
    context_repl = context.replace('TARGETWORD', lf_match)
    try:
        pre_idx = full_context.index(context_repl)
    except:
        try:
            pre_idx = full_context.index(context.split('TARGETWORD')[0].strip()[:-2])
        except:
            print('here')
    relevant_context = full_context[:pre_idx]
    # sectioned_text = list(filter(
    #     lambda x: len(x.strip()) > 0, re.split(header_regex, relevant_context.upper(), flags=re.M)))

    sep_symbol = ' headersep '
    sectioned_text = relevant_context.upper()
    for header_regex in header_regexes:
        sectioned_text = sep_symbol.join(re.split(header_regex, sectioned_text, flags=re.M))

    sectioned_tokens = list(filter(lambda x: len(x.strip()) > 0, sectioned_text.split(sep_symbol)))

    headers = []
    for tok_idx, toks in enumerate(sectioned_tokens):
        is_header = _is_header(sectioned_text, tok_idx)
        if is_header:
            header_stripped = toks.strip().strip(':').upper()
            if header_stripped in SECTION_NAMES:
                headers.append(create_section_token(header_stripped))
    if len(headers) == 0:
        return '<pad>'
    return headers[-1]


def tokenize_mimic_rs(text, header_regexes=None):
    tokenized_text = []
    sep_symbol = ' headersep '
    sectioned_text = text.upper()
    for header_regex in header_regexes:
        sectioned_text = sep_symbol.join(re.split(header_regex, sectioned_text, flags=re.M))

    sectioned_tokens = list(filter(lambda x: len(x.strip()) > 0, sectioned_text.split(sep_symbol)))
    for tok_idx, toks in enumerate(sectioned_tokens):
        if toks == ':':
            continue

        is_header = _is_header(sectioned_tokens, tok_idx)
        is_next_header = _is_header(sectioned_tokens, tok_idx + 1)

        if is_header and is_next_header:
            continue
        if is_header:
            header_stripped = toks.strip().strip(':').upper()
            if header_stripped in SECTION_NAMES:
                tokenized_text += [create_section_token(header_stripped)]
            else:
                toks = clean_text(toks)
                tokens = tokenize_str(toks)
                tokenized_text += tokens
        else:
            toks = clean_text(toks)
            tokens = tokenize_str(toks)
            tokenized_text += tokens
    tokenized_no_dup = []
    for i in range(len(tokenized_text)):
        if i == len(tokenized_text) - 1 or not tokenized_text[i] == tokenized_text[i + 1]:
            tokenized_no_dup.append(tokenized_text[i])
    return tokenized_no_dup


def unpack_section_names():
    fn = os.path.join('data', 'section_levels.json')
    if os.path.exists(fn):
        with open(fn, 'r') as fd:
            return json.load(fd)

    sn_len = np.array(list(map(len, SECTION_NAMES)))
    sn_order = np.argsort(-sn_len)

    levels = []
    for _ in range(10):
        levels.append(set())
    for idx in sn_order:
        section = SECTION_NAMES[idx]
        section = re.sub('\[\]', '', section)
        section = re.sub(r'\s+', ' ', section)
        for level_idx, level in enumerate(levels):
            any_matches = False
            for n in level:
                m = re.match(r'.+?' + section, n)
                if m and m.endpos == len(n):
                    any_matches = True
                    break
            if not any_matches:
                break
        levels[level_idx].add(section)
    level_list = []
    for level in levels:
        if len(level) > 0:
            level_list.append(list(level))
    return level_list


def preprocess_mimic_rs(window=10):
    with open('data/sf_lf_map.json', 'r') as fd:
        used_sf_lf_map = json.load(fd)

    mimic_df = pd.read_csv('../preprocess/data/mimic/NOTEEVENTS.csv')
    df = pd.read_csv('data/mimic_rs_dataset.csv')
    N = df.shape[0]
    tokenized_contexts = []
    sections = []
    trimmed_contexts = []

    section_levels = unpack_section_names()
    header_regexes = list(map(lambda level: r'\b({})(:)'.format('|'.join(level)), section_levels))
    print('Tokenizing and extracting context windows...')
    is_valid = []
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        context = row['context']
        tokens = tokenize_mimic_rs(context, header_regexes=header_regexes)
        is_header = list(map(lambda x: 'header=' in x, tokens))
        header_locs = np.where(np.array(is_header))[0]

        sf_idx = np.where(np.array(tokens) == 'targetword')[0]

        if not len(sf_idx) == 1:
            is_valid.append(False)
            tokenized_contexts.append(None)
            sections.append(None)
            trimmed_contexts.append(None)
            print('LF={} was tokenized out of context'.format(row['lf']))
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

            left_header = tokens[left_header_window] if left_header_window is not None else None
            left_header_boundary = 0 if left_header is None else left_header_window + 1

            if left_header is None:
                left_header = get_header_from_full_context(
                    mimic_df, context, row['lf_match'], row['doc_id'], header_regexes=header_regexes)

            right_header_boundary = len(tokens) if right_header_window is None else right_header_window

            left_window = max(sf_idx - window, 0, left_header_boundary)
            right_window = min(sf_idx + window + 1, len(tokens), right_header_boundary)
            window_tokens = tokens[left_window:sf_idx] + tokens[sf_idx + 1:right_window]

            if len(window_tokens) == 0:
                trimmed_contexts.append('<pad>')
            else:
                trimmed_contexts.append(' '.join(window_tokens))
            sections.append(left_header)
            tokenized_contexts.append(' '.join(tokens))
            is_valid.append(True)
        if (row_idx + 1) % 1000 == 0:
            print('Processed {} out of {} examples'.format(row_idx + 1, N))

    df['used_target_lf_idx'] = df['sf'].combine(df['lf_orig'], lambda sf, lf: used_sf_lf_map[sf].index(lf))
    df['row_idx'] = list(range(df.shape[0]))
    df.rename(columns={'lf': 'target_lf'}, inplace=True)
    df['trimmed_tokens'] = trimmed_contexts
    df['tokenized_context'] = tokenized_contexts
    df['section'] = sections
    df['is_valid'] = is_valid

    df = df[df['is_valid']]
    print('Removing {} invalid rows'.format(N - df.shape[0]))

    prev_N = df.shape[0]
    df.drop_duplicates(subset=['target_lf', 'tokenized_context'], inplace=True)
    N = df.shape[0]
    print('Removing {} duplicate rows. Saving {}'.format(prev_N - N, N))
    df.to_csv('data/mimic_rs_dataset_preprocessed_window_{}.csv'.format(window), index=False)


if __name__ == '__main__':
    preprocess_mimic_rs()
