import json

import pandas as pd
from collections import defaultdict


def _compute_marginal_probs(df, metadata_col):
    """
    :param df: reverse substitution dataset
    :param metadata_col: column in df representing metadata we care about
    :return: dictionary. For each acronym expansion (LF), returns metadata it is found in along with conditional
    probabilities p(metadata|LF)
    """
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
    """
    Computes empirical probabilities of p(LF|metadata) based on LF contexts extracted from MIMIC.
    These empirical counts are consumed by the LMC model when computing token marginals over metadata 
    """
    df = pd.read_csv('context_extraction/data/mimic_rs_preprocessed.csv')
    with open('data/metadata_marginals.json', 'w') as fd:
        json.dump(_compute_marginal_probs(df, 'metadata'), fd)
