from collections import defaultdict
import os

import numpy as np
import pandas as pd
import spacy
import torch

from compute_utils import compute_kl
from eval_utils import eval_tokenize, preprocess_minnesota_dataset


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


def evaluate_minnesota_acronyms(prev_args, model, vocab, mini=False, combine_phrases=False):
    chunker = None
    if combine_phrases:
        chunker = spacy.load('en_core_sci_sm')
    phrase_str = '_phrase' if combine_phrases else ''
    data_fp = 'eval_data/minnesota/preprocessed_dataset_window_{}{}.csv'.format(prev_args.window, phrase_str)
    if not os.path.exists(data_fp):
        print('Preprocessing Dataset...')
        preprocess_minnesota_dataset(prev_args.window, chunker=chunker, combine_phrases=combine_phrases)

    df = pd.read_csv(data_fp)

    if mini:
        df = df.sample(100, random_state=1992).reset_index(drop=True)

    context_ids = np.zeros([df.shape[0], prev_args.window * 2], dtype=int)
    center_ids = np.zeros([df.shape[0], ], dtype=int)
    ct = 0
    for row_idx, row in df.iterrows():
        assert row_idx == ct
        ct += 1
        row = row.to_dict()

        center_id = vocab.get_id(row['sf'].lower())
        center_ids[row_idx] = center_id
        context_id_seq = [vocab.get_id(token.lower()) for token in row['trimmed_tokens'].split()]
        context_ids[row_idx, :len(context_id_seq)] = context_id_seq

    center_id_tens = torch.LongTensor(center_ids).clamp_min_(0).to(prev_args.device)
    context_id_tens = torch.LongTensor(context_ids).clamp_min_(0).to(prev_args.device)
    # TODO add proper masking
    mask = torch.BoolTensor(torch.Size([center_id_tens.shape[0], context_id_tens.shape[-1]])).to(prev_args.device)
    mask.fill_(0)
    with torch.no_grad():
        z_mus, z_sigmas = model.encoder(center_id_tens, context_id_tens, mask)

    sense_fp = 'eval_data/minnesota/sense_inventory_ii'
    lfs, lf_sf_map, _ = parse_sense_df(sense_fp)

    tokenized_lfs = list(map(
        lambda t: eval_tokenize(t, unique_only=True, chunker=chunker, combine_phrases=combine_phrases), lfs))
    tokenized_ids = [list(map(vocab.get_id, tokens)) for tokens in tokenized_lfs]
    lf_priors = list(map(lambda x: model._compute_priors(torch.LongTensor(x).clamp_min_(0).to(prev_args.device)),
                         tokenized_ids))

    # for each SF, create matrix of LFs and priors
    sf_lf_prior_map = {}
    for lf, priors in zip(lfs, lf_priors):
        sf = lf_sf_map[lf]

        if sf not in sf_lf_prior_map:
            sf_lf_prior_map[sf] = {
                'lf': [],
                'lf_mu': [],
                'lf_sigma': []
            }

        sf_lf_prior_map[sf]['lf'].append(lf)
        # Representation of phrases is just average of words (...for now)
        sf_lf_prior_map[sf]['lf_mu'].append(priors[0].mean(axis=0).unsqueeze(0))
        sf_lf_prior_map[sf]['lf_sigma'].append(priors[1].mean(axis=0).unsqueeze(0))

    for sf in sf_lf_prior_map.keys():
        sf_lf_prior_map[sf]['lf_mu'] = torch.cat(sf_lf_prior_map[sf]['lf_mu'], axis=0)
        sf_lf_prior_map[sf]['lf_sigma'] = torch.cat(sf_lf_prior_map[sf]['lf_sigma'], axis=0)

    num_correct, target_kld, median_kld, num_eval = 0, 0.0, 0.0, 0
    random_expected_acc = 0.0
    for row_idx, row in df.iterrows():
        sf = row['sf']
        z_mu, z_sigma = z_mus[row_idx].unsqueeze(0), z_sigmas[row_idx].unsqueeze(0)
        lf_obj = sf_lf_prior_map[sf]
        lf_map, prior_mus, prior_sigmas = lf_obj['lf'], lf_obj['lf_mu'], lf_obj['lf_sigma']
        num_lf = prior_mus.shape[0]
        divergences = compute_kl(z_mu.repeat(num_lf, 1), z_sigma.repeat(num_lf, 1), prior_mus, prior_sigmas)

        closest_lf_idx = divergences.squeeze(-1).argmin().item()
        predicted_lf = lf_map[closest_lf_idx]
        target_lf = row['target_lf']

        try:
            target_lf_idx = lf_map.index(target_lf)
        except ValueError:
            target_lf_idx = -1
            for idx, lf in enumerate(lf_map):
                for lf_chunk in lf.split(';'):
                    if target_lf == lf_chunk:
                        target_lf_idx = idx
                        target_lf = lf_map[target_lf_idx]
                        break
        if target_lf in lf_map:
            num_eval += 1
            random_expected_acc += 1 / float(len(lf_map))
            target_kld += divergences[target_lf_idx].item()
            median_kld += divergences.median().item()
            if predicted_lf == target_lf:
                num_correct += 1

    N = float(num_eval)
    print('Statistics from {} examples'.format(int(N)))
    print('Minnesota Acronym Accuracy={}.  KLD(SF Posterior||Target LF)={}. Median KLD(SF Posterior || LF_(i..n)={}'.
          format(num_correct / N, target_kld / N, median_kld / N))
    print('Random Accuracy={}'.format(random_expected_acc / N))


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
