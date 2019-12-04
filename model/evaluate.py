import pickle
import sys

import argparse
from scipy.spatial.distance import cosine
import torch

from model_utils import restore_model, tensor_to_np
from vae import VAE


sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')


def point_similarity(model, vocab, token_a, token_b):
    id_a = vocab.get_id(token_a)
    id_b = vocab.get_id(token_b)

    assert id_a >= 0 and id_b >= 0
    embeddings = tensor_to_np(model.embeddings_mu.weight)
    sim = 1.0 - cosine(embeddings[id_a, :], embeddings[id_b, :])
    print('Cosine Similarity between {} and {} --> {}'.format(token_a, token_b, sim))
    return cosine


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('--eval_fp', default='../preprocess/data/')
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')

    args = parser.parse_args()

    # Load Vocabulary
    vocab_infile = '../preprocess/data/vocab.pk'
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    vocab_size = vocab.size()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    prev_args, vae_model, optimizer_state = restore_model(vocab_size, args.experiment)

    # Make sure it's NOT calculating gradients
    vae_model = vae_model.to(args.device)
    vae_model.eval()  # just sets .requires_grad = False

    point_similarity(vae_model, vocab, 'advil', 'tylenol')
    point_similarity(vae_model, vocab, 'advil', 'tolerate')
    point_similarity(vae_model, vocab, 'advil', 'pain')
    point_similarity(vae_model, vocab, 'advil', 'building')
    point_similarity(vae_model, vocab, 'advil', 'lawyer')

    import os
    import numpy as np
    print('Loading data...')
    ids_infile = os.path.join('../preprocess/data/', 'ids.npy')
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)
    num_tokens = len(ids)

    advil_id = vocab.get_id('advil')
    tylenol_id = vocab.get_id('tylenol')
    building_id = vocab.get_id('building')

    advil_locs = np.where(ids == advil_id)[0]
    tylenol_locs = np.where(ids == tylenol_id)[0]
    building_locs = np.where(ids == building_id)[0]

    for i in range(10):
        center = advil_locs[i]
        window_ids = ids[center - 5: center + 5]
        window_toks = [vocab.get_token(id) for id in window_ids]
        print(' '.join(window_toks))

    print('\n\n\n\n')

    for i in range(10):
        center = tylenol_locs[i]
        window_ids = ids[center - 5: center + 5]
        window_toks = [vocab.get_token(id) for id in window_ids]
        print(' '.join(window_toks))

    print('\n\n\n\n\n')
    for i in range(10):
        center = building_locs[i]
        window_ids = ids[center - 5: center + 5]
        window_toks = [vocab.get_token(id) for id in window_ids]
        print(' '.join(window_toks))
