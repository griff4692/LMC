import pickle
import sys

import argparse
import torch

from model_utils import restore_model
from vae import VAE


sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')


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
