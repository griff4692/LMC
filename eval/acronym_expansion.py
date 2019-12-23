from collections import defaultdict

import pandas as pd
import torch

from compute_utils import compute_kl


def parse_sense_df(sense_fp):
    sense_df = pd.read_csv(sense_fp, sep='|')
    sense_df.dropna(subset=['LF', 'SF'], inplace=True)
    lfs = sense_df['LF'].unique().tolist()
    lf_sf_map = {}
    sf_lf_map = defaultdict(set)
    for row_idx, row in sense_df.iterrows():
        row = row.to_dict()
        lf_sf_map[row['LF']] = row['SF']
        sf_lf_map[row['SF']].add(row['LF'])
    for k in sf_lf_map:
        sf_lf_map[k] = list(sorted(sf_lf_map[k]))
    return lfs, lf_sf_map, sf_lf_map


# TODO deprecated for now
def evaluate_mimic_acronyms(prev_args, model, vocab):
    print('Evaluating context-dependent representations via acronym disambiguation...')
    label_df = pd.read_csv('eval_data/mimic_acronym_expansion_labels.csv')
    expansion_df = pd.read_csv('eval_data/acronym_expansions.csv')
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

        center_id_tens = torch.LongTensor([center_id]).clamp_min_(0).to(prev_args.device)
        context_id_tens = torch.LongTensor(context_id_seq).unsqueeze(0).clamp_min_(0).to(prev_args.device)

        mask = torch.BoolTensor(torch.Size([1, len(context_id_seq)])).to(prev_args.device)
        mask.fill_(0)

        p_mu, p_sigma = model._compute_priors(center_id_tens)
        min_prior_kl, min_posterior_kl = float('inf'), float('inf')

        with torch.no_grad():
            z_mu, z_sigma = model.encoder(center_id_tens, context_id_tens, mask)

        for target_expansion in target_lfs.split('|'):
            lf_ids = [vocab.get_id(token.lower()) for token in target_expansion.split()]
            lf_tens = torch.LongTensor(lf_ids).clamp_min_(0).to(prev_args.device)

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
