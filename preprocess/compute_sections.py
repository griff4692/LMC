from collections import Counter, defaultdict
from multiprocessing import Pool
import os
import re
from time import time

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][A-z0-9/ ]+[A-z]:)'


def enumerate_metadata_ids_multi_bsg(ids, sec_pos_idxs, cat_pos_idxs):
    sec_ids = []
    cat_ids = []

    sec_pointer = 0
    cat_pointer = 0
    N = len(ids)

    num_sec = len(sec_pos_idxs)
    num_cat = len(cat_pos_idxs)

    curr_cat_id = 0
    while sec_pointer < num_sec and cat_pointer < num_cat:
        sec_pos_idx = sec_pos_idxs[sec_pointer]
        cat_pos_idx = cat_pos_idxs[cat_pointer]

        next_sec_pos_idx = sec_pos_idxs[sec_pointer + 1] if sec_pointer + 1 < num_sec else N
        next_cat_pos_idx = cat_pos_idxs[cat_pointer + 1] if cat_pointer + 1 < num_cat else N

        min_pos_idx = min(sec_pos_idx, cat_pos_idx)
        max_pos_idx = max(sec_pos_idx, cat_pos_idx)

        metadata_len = min(max_pos_idx, next_sec_pos_idx, next_cat_pos_idx) - min_pos_idx
        id = ids[min_pos_idx]
        if sec_pos_idx < cat_pos_idx:  # We are processing a section
            sec_pointer += 1
            curr_sec_id = id
        else:  # We are processing the document category
            cat_pointer += 1
            curr_sec_id = 0
            curr_cat_id = id

        sec_ids += [curr_sec_id] * metadata_len
        cat_ids += [curr_cat_id] * metadata_len

    for remaining_sec_pointer in range(sec_pointer, num_sec):
        next_sec_pos_idx = sec_pos_idxs[remaining_sec_pointer + 1] if remaining_sec_pointer + 1 < num_sec else N
        sec_pos_idx = sec_pos_idxs[remaining_sec_pointer]
        metadata_len = next_sec_pos_idx - sec_pos_idx
        curr_sec_id = ids[sec_pos_idx]
        sec_ids += [curr_sec_id] * metadata_len
        cat_ids += [curr_cat_id] * metadata_len

    for remaining_cat_pointer in range(cat_pointer, num_cat):
        next_cat_pos_idx = cat_pos_idxs[remaining_cat_pointer + 1] if remaining_cat_pointer + 1 < num_cat else N
        cat_pos_idx = cat_pos_idxs[remaining_cat_pointer]
        metadata_len = next_cat_pos_idx - cat_pos_idx
        curr_cat_id = ids[cat_pos_idx]
        sec_ids += [0] * metadata_len
        cat_ids += [curr_cat_id] * metadata_len

    assert len(sec_ids) == len(cat_ids) == N
    return sec_ids, cat_ids


def enumerate_metadata_ids_lmc(ids, metadata_pos_idxs, token_vocab, metadata_vocab):
    """
    :param ids: List of token ids.
    :param metadata_pos_idxs: Positional indices where ids are metadata
    :return: List of len(ids) representing the metadata each token in ids lies within (for efficient access by batcher)
    """
    metadata_ids = [0] * metadata_pos_idxs[0]
    token_metadata_counts = defaultdict(list)
    for metadata_num, metadata_pos_idx in tqdm(enumerate(metadata_pos_idxs), total=len(metadata_pos_idxs)):
        token_metadata_id = ids[metadata_pos_idx]
        # Make an adjustment to find the appropriate id in the new metadata vocabulary
        metadata_id = metadata_vocab.get_id(token_vocab.get_token(token_metadata_id))
        assert metadata_id >= 0
        if metadata_num + 1 == len(metadata_pos_idxs):
            metadata_len = len(ids) - metadata_pos_idx
        else:
            metadata_len = metadata_pos_idxs[metadata_num + 1] - metadata_pos_idx
        metadata_ids += [metadata_id] * metadata_len
        for id in list(ids[metadata_pos_idx + 1:metadata_pos_idx + metadata_len]):
            token_metadata_counts[id] += [metadata_id]
    for k, v in token_metadata_counts.items():
        tf = Counter(v)
        keys = list(tf.keys())
        freqs = []
        for key in keys:
            freqs.append(tf[key])
        freqs = np.array(freqs, dtype=float)
        token_metadata_counts[k] = (keys, freqs)
    return metadata_ids, token_metadata_counts


def extract_headers(text):
    matches = re.findall(HEADER_SEARCH_REGEX, text, re.M)
    return list(map(lambda match: match.upper().strip().strip(':'), matches))


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--mimic_fp', default='data/mimic/NOTEEVENTS')
    arguments.add_argument('-debug', default=False, action='store_true')

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    debug_str = '_mini' if args.debug else ''
    df = pd.read_csv('{}{}.csv'.format(args.mimic_fp, debug_str))

    texts = df['TEXT'].tolist()

    print('Extracting section headers...')
    start_time = time()
    p = Pool()
    sections = p.map(extract_headers, texts)
    p.close()
    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))

    print('Done!  Now extracting counts...')
    all_counts = 0
    section_counts = defaultdict(int)
    for section_arr in sections:
        for section in section_arr:
            section_counts[section] += 1
            all_counts += 1
    print('Located {} section headers in all {} documents'.format(all_counts, len(texts)))

    df = pd.DataFrame(section_counts.items(), columns=['section', 'count'])
    df.to_csv('data/mimic/section{}.csv'.format(debug_str), index=False)
