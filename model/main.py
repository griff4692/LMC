import pickle
import os
from shutil import rmtree
import sys
from time import sleep

import argparse
import numpy as np
import torch
from tqdm import tqdm

from batcher import SkipGramBatchLoader
from model_utils import restore_model
from vae import VAE
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')
    parser.add_argument('--restore_experiment', default=None, help='Experiment name from which to restore.')

    # Training Hyperparameters
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--window', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    # Model Hyperparameters
    parser.add_argument('--encoder_hidden_dim', default=64, type=int, help='hidden dimension for encoder')
    parser.add_argument('--encoder_input_dim', default=64, type=int, help='embedding dimemsions for encoder')
    parser.add_argument('--hinge_loss_margin', default=1.0, type=float, help='reconstruction margin')
    parser.add_argument('--latent_dim', default=200, type=int, help='z dimension')

    args = parser.parse_args()

    # Load Data
    debug_str = '_mini' if args.debug else ''
    ids_infile = os.path.join(args.data_dir, 'ids{}.npy'.format(debug_str))
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)
    num_tokens = len(ids)
    # The document boundary index (pad_idx = 0) is never going to be a center word
    ignore_idxs = np.where(ids == 0)[0]
    # Load Vocabulary
    vocab_infile = '../preprocess/data/vocab{}.pk'.format(debug_str)
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    vocab_size = vocab.size()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_str)
    print('Training on {}...'.format(device_str))

    batcher = SkipGramBatchLoader(num_tokens, ignore_idxs, batch_size=args.batch_size)

    vae_model = VAE(args, vocab_size).to(args.device)
    checkpoint_state = None
    if args.restore_experiment is not None:
        restore_model(vae_model, vocab_size, args.restore_experiment)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, vae_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    if checkpoint_state is not None:
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)

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
            center_ids_tens = torch.LongTensor(center_ids).to(args.device)
            context_ids_tens = torch.LongTensor(context_ids).to(args.device)

            neg_ids = np.random.choice(vocab.size(), size=context_ids_tens.shape)
            neg_ids_tens = torch.LongTensor(vocab.neg_sample(size=context_ids_tens.shape)).to(args.device)

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

        # Serializing everything from model weights and optimizer state, to to loss function and arguments
        losses_dict = {'losses': {'joint': epoch_joint_loss, 'kl': epoch_kl_loss, 'recon': epoch_recon_loss}}
        state_dict = {'model_state_dict': vae_model.state_dict()}
        state_dict.update(losses_dict)
        state_dict.update({'optimizer_state_dict': optimizer.state_dict()})
        args_dict = {'args': {arg: getattr(args, arg) for arg in vars(args)}}
        state_dict.update(args_dict)
        state_dict.update({'vocab': vocab})
        # Serialize model and statistics
        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        print('Saving model state to {}'.format(checkpoint_fp))
        torch.save(state_dict, checkpoint_fp)
