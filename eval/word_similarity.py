import os

import pandas as pd
from scipy.stats import spearmanr, pearsonr

from eval_utils import point_similarity, w2v_point_similarity


def evaluate_word_similarity(model, vocab, data_dir='eval_data', gensim_w2v=False):
    umnrs = {
        'name': 'UMNRS',
        'file': os.path.join(data_dir, 'UMNSRS_relatedness.csv'),
        'label': 'Mean',
        't1': 'Term1',
        't2': 'Term2',
    }

    mayo = {
        'name': 'MayoSRS',
        'file': os.path.join(data_dir, 'MayoSRS.csv'),
        'label': 'Mean',
        't1': 'TERM1',
        't2': 'TERM2',
    }

    sim_datasets = [umnrs, mayo]

    for didx, sim_dataset in enumerate(sim_datasets):
        word_sim_df = pd.read_csv(sim_dataset['file'])
        human_scores = word_sim_df[sim_dataset['label']].tolist()
        known_model_relatedness, known_human_scores = [], []
        for row_idx, row in word_sim_df.iterrows():
            row = row.to_dict()
            t1 = row[sim_dataset['t1']].lower()
            t2 = row[sim_dataset['t2']].lower()
            if gensim_w2v:
                sim = w2v_point_similarity(model, t1, t2)
            else:
                sim = point_similarity(model, vocab, t1, t2)
            if not sim == 0.0:  # means both terms are OOV
                known_human_scores.append(human_scores[row_idx])
                known_model_relatedness.append(sim)
        pear_corr, _ = pearsonr(known_model_relatedness, known_human_scores)
        spear_corr, _ = spearmanr(known_model_relatedness, known_human_scores)
        sim_datasets[didx]['pearson_correlation'] = pear_corr
        sim_datasets[didx]['spearman_correlation'] = spear_corr
        print('{} Evaluation\n\tWord Similarity --> Pearson Corr.={}, Spearman Corr.={}'.format(
            sim_dataset['name'], pear_corr, spear_corr))
    return sim_datasets
