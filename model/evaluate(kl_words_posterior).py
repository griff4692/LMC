import sys

import argparse
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import torch
import numpy as np
from model_utils import restore_model, tensor_to_np
from vae import VAE
import pickle
from batcher import SkipGramBatchLoader
from compute_utils import mask_2D



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
    word_sim_df = pd.read_csv('eval_data//MayoSRS.csv')
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
    
def kl_similarity_prior(model, vocab, tokens_a, tokens_b):
    
    ids_a = get_known_ids(vocab, tokens_a)
    ids_b = get_known_ids(vocab, tokens_b)



    sigma = tensor_to_np(model.embeddings_log_sigma.weight.exp())
    mu =  tensor_to_np(model.embeddings_mu.weight)
    
    sigma_a = sigma[ids_a, :].mean(0)
    sigma_b = sigma[ids_b, :].mean(0)
    
    mu_a = mu[ids_a, :].mean(0)
    mu_b = mu[ids_b, :].mean(0)
    
    sigma_b_inv = sigma_b**-1
    d = mu_a.shape[0]
    mu_diff = mu_a-mu_b
    dot = np.dot(mu_diff,mu_diff)
    kl_div = 0.5*(d*np.log(sigma_b/sigma_a)-d+d*sigma_b_inv*sigma_a+dot*sigma_b_inv)
                  
    return kl_div[0]

def kl_similarity_posterior(model,vocab,tokens_a,tokens_b):
    
    window_size = 5
    with open("ids.npy", 'rb') as fd:
        ids = np.load(fd)
    num_tokens = len(ids)
    
    ignore_idxs = np.where(ids == 0)[0]
    # Load Vocabulary
    
    ids_a = get_known_ids(vocab, tokens_a)
    ids_b = get_known_ids(vocab, tokens_b)
    
    
    num_context_ids = window_size*2
    
    batch_size = 1
    
    num_contexts = batch_size
    
    batcher = SkipGramBatchLoader(num_tokens, ignore_idxs, batch_size=batch_size)
    
    context_ids_a = batcher.extract_context_ids(ids,ids_a[0],window_size)
    context_ids_b = batcher.extract_context_ids(ids,ids_b[0],window_size)
    
    
    mask_size = torch.Size([batch_size, num_context_ids])
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    mask_a = mask_2D(mask_size, [1]).to(device)
    mask_b = mask_2D(mask_size, [1]).to(device)
    
    pos_mu_a,pos_sig_a = model.encoder(torch.tensor(ids_a).to(device), torch.tensor(context_ids_a).to(torch.int64).to(device).view(1,10), mask_a)
    pos_mu_b,pos_sig_b = model.encoder(torch.tensor(ids_b).to(device), torch.tensor(context_ids_b).to(torch.int64).to(device).view(1,10), mask_b)
    
    mu_a = pos_mu_a.cpu().data.numpy().reshape(-1,)
    mu_b = pos_mu_b.cpu().data.numpy().reshape(-1,)
    sigma_a = pos_sig_a.cpu().data.numpy()[0][0]
    sigma_b = pos_sig_b.cpu().data.numpy()[0][0]
    
    sigma_b_inv = sigma_b**-1
    d = mu_a.shape[0]
    mu_diff = mu_a-mu_b
    dot = np.dot(mu_diff,mu_diff)
    kl_div = 0.5*(d*np.log(sigma_b/sigma_a)-d+d*sigma_b_inv*sigma_a+dot*sigma_b_inv)
    
    return kl_div




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

    print('lesion', 'nodule', kl_similarity_posterior(model, vocab, 'lesion', 'nodule'))
    print('lesion', 'advil', kl_similarity_posterior(model, vocab, 'lesion', 'advil'))
    print('pain', 'tolerance', kl_similarity_posterior(model, vocab, 'pain', 'tolerance'))
    print('advil', 'tylenol', kl_similarity_posterior(model, vocab, 'advil', 'tylenol'))
    print('advil', 'ibruprofen', kl_similarity_posterior(model, vocab, 'advil', 'ibruprofen'))
    print('headache', 'migraine', kl_similarity_posterior(model, vocab, 'headache', 'migraine'))
    print('tolerance', 'lesion', kl_similarity_posterior(model, vocab, 'tolerance', 'lesion'))
