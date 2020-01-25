import json

import pandas as pd
from collections import defaultdict


def _compute_marginal_probs(df, metadata_col):
    freqs = df.groupby('target_lf_sense')[metadata_col].value_counts().to_dict()
    lf_marginals = defaultdict(lambda: {metadata_col: [], 'count': []})
    for (target_lf_sense, metadata), count in freqs.items():
        lf_marginals[target_lf_sense][metadata_col].append(metadata)
        lf_marginals[target_lf_sense]['count'].append(count)
    for lf, vals in lf_marginals.items():
        normalizer = float(sum(lf_marginals[lf]['count']))
        p = list(map(lambda x: x / normalizer, lf_marginals[lf]['count']))
        lf_marginals[lf]['p'] = p

    return lf_marginals


if __name__ == '__main__':
    df = pd.read_csv('data/mimic_rs_dataset_preprocessed_window_10.csv')
    lf_cat_marginals = _compute_marginal_probs(df, 'category')
    lf_sec_marginals = _compute_marginal_probs(df, 'section')

    with open('data/category_marginals.json', 'w') as fd:
        json.dump(lf_cat_marginals, fd)

    with open('data/section_marginals.json', 'w') as fd:
        json.dump(lf_sec_marginals, fd)
