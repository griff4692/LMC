from collections import Counter, defaultdict
from multiprocessing import Pool
import os
import re
from time import time

import argparse
import numpy as np
import pandas as pd


HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][A-z0-9/ ]+[A-z]:)'


def enumerate_metadata_ids_lmc(ids, metadata_pos_idxs, token_vocab, metadata_vocab):
    """
    :param ids: List of token ids.
    :param metadata_pos_idxs: Positional indices where ids are metadata
    :return: List of len(ids) representing the metadata each token in ids lies within (for efficient access by batcher)
    """
    metadata_ids = [0] * metadata_pos_idxs[0]
    token_metadata_counts = defaultdict(list)
    for metadata_num, metadata_pos_idx in enumerate(metadata_pos_idxs):
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
        freqs /= freqs.sum()
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
