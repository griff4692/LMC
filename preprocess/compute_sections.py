from collections import Counter, defaultdict
import os
import re

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


"""
Deprecated for now - TODO remove
"""
def enumerate_section_ids(ids, section_pos_idxs):
    """
    :param ids: List of token ids where negative numbers are section ids and 0 are document boundaries.
    Regular tokens are represented by positive ids
    :param section_pos_idxs: Positional indices where ids <= 0 indicating either section or document boundaries
    :return: List of len(ids) representing the section each token in ids lies within (for efficient access by batcher)
    """
    raise Exception('Not supported!')
    # section_ids = []
    # for section_num, section_pos_idx in enumerate(section_pos_idxs):
    #     section_id = ids[section_pos_idx]
    #     assert section_id >= 0
    #     if section_num + 1 == len(section_pos_idxs):
    #         section_len = len(ids) - section_pos_idx
    #     else:
    #         section_len = section_pos_idxs[section_num + 1] - section_pos_idx
    #     section_ids += [section_id] * section_len
    # return section_ids


def enumerate_metadata_ids_lmc(ids, metadata_pos_idxs, token_vocab, metadata_vocab):
    """
    :param ids: List of token ids.
    :param metadata_pos_idxs: Positional indices where ids are metadata
    :return: List of len(ids) representing the metadata each token in ids lies within (for efficient access by batcher)
    """
    metadata_ids = [-1] * metadata_pos_idxs[0]
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

    sections = defaultdict(int)
    texts = df['TEXT'].tolist()

    for i in tqdm(range(len(texts))):
        pattern = "^([A-Z][A-z0-9 ]+[A-z])(:)"
        matches = re.findall(pattern, texts[i], re.M)

        for match in matches:
            sections[match[0]] += 1

    df_arr = []
    for section in sections:
        count = sections[section]
        df_arr.append((section, sections[section]))
    df = pd.DataFrame(df_arr, columns=['section', 'count'])
    df.to_csv('data/mimic/sections.csv', index=False)
