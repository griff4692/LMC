from collections import defaultdict
import sys

import argparse
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import torch

from compute_utils import compute_kl
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


def evaluate_acronyms(prev_args, model, vocab):
    print('Evaluating context-dependent representations via acronym disambiguation...')
    label_df = pd.read_csv('../eval_data/mimic_acronym_expansion_labels.csv')
    expansion_df = pd.read_csv('../eval_data/acronym_expansions.csv')
    expansion_df = expansion_df[expansion_df['lf_count'] > 0]

    target_window_size = prev_args.window

    sf_lf_map = defaultdict(set)
    for row_idx, row in expansion_df.iterrows():
        row = row.to_dict()
        sf_lf_map[row['sf']].add(row['lf'])

    contexts = label_df['context'].tolist()
    context_ids = []
    for row_idx, context in enumerate(contexts):
        sc = context.split()
        for cidx in range(len(sc)):
            if sc[cidx] == 'TARGETWORD':
                break

        left_idx = max(0, cidx - target_window_size)
        right_idx = min(cidx + target_window_size + 1, len(sc))
        context_tokens = sc[left_idx:cidx] + sc[cidx + 1: right_idx]
        context_ids.append([
            vocab.get_id(token.lower()) for token in context_tokens
        ])

    prior_kls = []
    posterior_kls = []

    for row_idx, row in label_df.iterrows():
        row = row.to_dict()
        sf, target_lfs = row['sf'], row['lf']
        context_id_seq = context_ids[row_idx]
        center_id = vocab.get_id(sf.lower())

        center_id_tens = torch.LongTensor([center_id]).to(prev_args.device).clamp_min_(0)
        context_id_tens = torch.LongTensor(context_id_seq).unsqueeze(0).to(prev_args.device).clamp_min_(0)

        mask = torch.BoolTensor(torch.Size([1, len(context_id_seq)])).to(prev_args.device)
        mask.fill_(0)

        p_mu, p_sigma = model._compute_priors(center_id_tens)
        min_prior_kl, min_posterior_kl = float('inf'), float('inf')

        with torch.no_grad():
            z_mu, z_sigma = model.encoder(center_id_tens, context_id_tens, mask)

        for target_expansion in target_lfs.split('|'):
            lf_ids = [vocab.get_id(token.lower()) for token in target_expansion.split()]
            lf_tens = torch.LongTensor(lf_ids).to(prev_args.device).clamp_min_(0)

            lf_mu, lf_sigma = model._compute_priors(lf_tens)

            avg_lf_mu = lf_mu.mean(axis=0)
            avg_lf_sigma = lf_sigma.mean(axis=0)

            prior_kl = compute_kl(p_mu, p_sigma, avg_lf_mu, avg_lf_sigma).item()
            posterior_kl = compute_kl(z_mu, z_sigma, avg_lf_mu, avg_lf_sigma).item()

            min_prior_kl = min(prior_kl, min_prior_kl)
            min_posterior_kl = min(posterior_kl, min_posterior_kl)
        prior_kls.append(min_prior_kl)
        posterior_kls.append(min_posterior_kl)

    avg_prior_kl = sum(prior_kls) / float(len(prior_kls))
    avg_posterior_distances = sum(posterior_kls) / float(len(posterior_kls))
    print('Avg Prior KLD={}. Avg Posterior KLD={}'.format(avg_prior_kl, avg_posterior_distances))


def evaluate_word_similarity(model, vocab):
    umnrs = {
        'name': 'UMNRS',
        'file': '../eval_data/UMNSRS_relatedness.csv',
        'label': 'Mean',
        't1': 'Term1',
        't2': 'Term2',
    }

    mayo = {
        'name': 'MayoSRS',
        'file': '../eval_data/MayoSRS.csv',
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
    evaluate_acronyms(prev_args, model, vocab)
    word_sim_results = evaluate_word_similarity(model, vocab)
