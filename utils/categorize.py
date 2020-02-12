# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:12:56 2020

@author: Mert Ketenci
"""
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def categorize_token_metadata_counts(token_metadata_counts,metadata_vocab,category_dict):
    for key in tqdm(token_metadata_counts.keys()):
        category_count = defaultdict(int)
        categories = []
        for i in range(len(token_metadata_counts[key][0])):
            document = token_metadata_counts[key][0][i]
            for category in category_dict[metadata_vocab.get_token(document)]:
                category_count[category] += token_metadata_counts[key][1][i]
            categories.append(category_dict[metadata_vocab.get_token(document)])
        categories = list(set([item for sublist in categories for item in sublist]))
        token_metadata_counts[key] = list(token_metadata_counts[key])
        token_metadata_counts[key][0] = categories
        token_metadata_counts[key][1] = np.asarray([category_count[category] for category in categories])
        token_metadata_counts[key] = tuple(token_metadata_counts[key])
    return token_metadata_counts