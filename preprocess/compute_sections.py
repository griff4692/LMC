from collections import Counter, defaultdict
from multiprocessing import Pool
import re
from time import time

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][-A-z0-9/() ]{0,100}[A-z0-9]:)'


def enumerate_metadata_ids(ids, metadata_pos_idxs):
    """
    :param ids: List of token ids.
    :param metadata_pos_idxs: Positional indices where ids are metadata
    :return: metadata_ids of length(ids).  The ith item in meta_ids
    signals the metadata vocabulary id from which the token at position index i belongs.
    """
    N = len(ids)
    meta_ids = []

    num_metadata = len(metadata_pos_idxs)
    for idx, meta_pos_idx in tqdm(enumerate(metadata_pos_idxs)):
        meta_id = ids[meta_pos_idx]
        end_idx = metadata_pos_idxs[idx + 1] if idx + 1 < num_metadata else N
        meta_ids[meta_pos_idx:end_idx] = [meta_id] * (end_idx - meta_pos_idx)
    return meta_ids


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
    """
    :param text: string
    :return: a list of matched, uppercased section header names
    e.g.: text = HPI: this patient was previously treated for same illness. Social History : smoker
    return output is [HPI, SOCIAL HISTORY]
    """
    matches = re.findall(HEADER_SEARCH_REGEX, text, re.M)
    return list(map(lambda match: match.upper().strip().strip(':'), matches))
