import json
import pickle
import os
from time import sleep

import argparse
import numpy as np
import torch
from tqdm import tqdm

from model.batcher import SkipGramBatchLoader
from model.vae import VAE
from preprocess.vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')

    # Training Hyperparameters
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--window', default=5, type=int)

    # Model Hyperparameters
    parser.add_argument('--latent_dim', default=100, type=int, help='z dimension')
    parser.add_argument('--encoder_input_dim', default=50, type=int, help='embedding dimemsions for encoder')
    parser.add_argument('--encoder_hidden_dim', default=50, type=int, help='hidden dimension for encoder')

    args = parser.parse_args()

    # FIX THIS FOR NOW
    args.debug = True

    # Load Data
    debug_str = '_mini' if args.debug else ''
    ids_infile = os.path.join(args.data_dir, 'ids{}.npy'.format(debug_str))
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)
    num_tokens = len(ids)
    # the document boundary index (pad_idx=0) is never going to be a center word
    ignore_idxs = np.where(ids == 0)[0]
    # Load Vocabulary
    vocab_infile = '../preprocess/data/vocab{}.pk'.format(debug_str)
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    vocab_size = vocab.size()

    batcher = SkipGramBatchLoader(num_tokens, ignore_idxs)
    vae_model = VAE(args, vocab_size)

    trainable_params = filter(lambda x: x.requires_grad, vae_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)

    # Make sure it's calculating gradients
    vae_model.train()  # just sets .requires_grad = True
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        batcher.reset()
        num_batches = batcher.num_batches()
        epoch_joint_loss, epoch_kl_loss, epoch_recon_loss = 0.0, 0.0, 0.0
        for _ in tqdm(range(num_batches)):
            # Reset gradients
            optimizer.zero_grad()

            center_ids, context_ids = batcher.next(ids, args.window)
            center_ids_tens = torch.LongTensor(center_ids)
            context_ids_tens = torch.LongTensor(context_ids)
            neg_ids_tens = torch.ones_like(context_ids_tens).long()

            kl_loss, recon_loss = vae_model(center_ids_tens, context_ids_tens, neg_ids_tens)
            joint_loss = kl_loss + recon_loss
            joint_loss.backward()  # backpropagate loss

            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_joint_loss += joint_loss.item()
            optimizer.step()
        epoch_joint_loss /= float(batcher.num_batches())
        epoch_kl_loss /= float(batcher.num_batches())
        epoch_recon_loss /= float(batcher.num_batches())
        sleep(0.1)
        print('Epoch={}. Joint loss={}.  KL Loss={}. Reconstruction Loss={}'.format(
            epoch, epoch_joint_loss, epoch_kl_loss, epoch_recon_loss))
        assert not batcher.has_next()
