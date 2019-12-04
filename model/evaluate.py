import sys

import argparse
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import torch

from model_utils import restore_model, tensor_to_np
from vae import VAE


sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')


def get_known_ids(vocab, tokens):
    return list(filter(lambda id: id > -1, list(map(lambda tok: vocab.get_id(tok), tokens.split(' ')))))


def point_similarity(model, vocab, tokens_a, tokens_b):
    ids_a = get_known_ids(vocab, tokens_a)
    ids_b = get_known_ids(vocab, tokens_b)

    if len(ids_a) == 0 or len(ids_b) == 0:
        return 0.0

    embeddings = tensor_to_np(model.embeddings_mu.weight)

    rep_a = embeddings[ids_a, :].mean(0)
    rep_b = embeddings[ids_b, :].mean(0)
    sim = 1.0 - cosine(rep_a, rep_b)
    return sim


def evaluate_word_similarity(model, vocab):
    word_sim_df = pd.read_csv('../eval_data/MayoSRS.csv')
    human_scores = word_sim_df['Mean'].tolist()
    known_model_relatedness, known_human_scores = [], []
    for row_idx, row in word_sim_df.iterrows():
        row = row.to_dict()
        t1 = row['TERM1']
        t2 = row['TERM2']
        sim = point_similarity(model, vocab, t1, t2)
        if not sim == 0.0:
            known_human_scores.append(human_scores[row_idx])
            known_model_relatedness.append(sim)
    pear_corr, _ = pearsonr(known_model_relatedness, known_human_scores)
    spear_corr, _ = spearmanr(known_model_relatedness, known_human_scores)
    return pear_corr, spear_corr


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('--eval_fp', default='../preprocess/data/')
    parser.add_argument('--experiment', default='debug', help='Save path in weights/ for experiment.')

    args = parser.parse_args()

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    prev_args, vae_model, vocab, optimizer_state = restore_model(args.experiment)

    # Make sure it's NOT calculating gradients
    model = vae_model.to(args.device)
    model.eval()  # just sets .requires_grad = False

    print('\nEvaluations...')
    pear_corr, spear_corr = evaluate_word_similarity(model, vocab)
    print('MayoSRS Evaluation\n\tWord Similarity --> Pearson Corr.={}, Spearman Corr.={}'.format(pear_corr, spear_corr))

    point_similarity(model, vocab, 'constipation', 'diarrhea')
    point_similarity(model, vocab, 'diarrhea', 'parking')
    point_similarity(model, vocab, 'melanoma', 'brain')
    point_similarity(model, vocab, 'garage', 'parking')
