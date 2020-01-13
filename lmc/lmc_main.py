import pandas as pd
import pickle
import os
import re
from shutil import rmtree
import sys
from time import sleep

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_sections import enumerate_section_ids_lmc
from lmc_utils import restore_model, save_checkpoint
from lmc_batcher import SkipGramBatchLoader
from lmc_model import LMC
from model_utils import get_git_revision_hash, render_args
from vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')
    parser.add_argument('--restore_experiment', default=None, help='Experiment name from which to restore.')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('-combine_phrases', default=False, action='store_true')
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--window', default=5, type=int)

    # Model Hyperparameters
    parser.add_argument('--encoder_hidden_dim', default=64, type=int, help='hidden dimension for encoder')
    parser.add_argument('--encoder_input_dim', default=100, type=int, help='embedding dimemsions for encoder')
    parser.add_argument('--hinge_loss_margin', default=1.0, type=float, help='reconstruction margin')
    parser.add_argument('--latent_dim', default=100, type=int, help='z dimension')
    parser.add_argument('-sample_sections', action='store_true', default=False,
                        help='Sample 1 section rather than compute full marginal across all sections.')

    args = parser.parse_args()
    args.git_hash = get_git_revision_hash()
    render_args(args)

    # Load Data
    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''

    ids_infile = os.path.join(args.data_dir, 'ids{}{}.npy'.format(debug_str, phrase_str))
    print('Loading data from {}...'.format(ids_infile))
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)

    # Load Vocabulary
    vocab_infile = '../preprocess/data/vocab{}{}.pk'.format(debug_str, phrase_str)
    print('Loading vocabulary from {}...'.format(vocab_infile))
    with open(vocab_infile, 'rb') as fd:
        token_vocab = pickle.load(fd)
    print('Loaded vocabulary of size={}...'.format(token_vocab.separator_start_vocab_id))

    section_counts = pd.read_csv('../preprocess/data/mimic/sections.csv')
    section_counts['header'] = section_counts['section'].apply(lambda x: 'header=' + re.sub(r'\s+', '', x.upper()))
    section_counts_dict = section_counts.set_index('header').to_dict()['count']
    section_counts_dict['header=DOCUMENT'] = 0

    print('Collecting document information...')
    section_pos_idxs = np.where(ids <= 0)[0]
    section_id_range = np.arange(token_vocab.separator_start_vocab_id, token_vocab.size())
    section_vocab = Vocab()
    for section_id in section_id_range:
        section_name = token_vocab.get_token(section_id)
        section_vocab.add_token(section_name, token_support=section_counts_dict[section_name])
    full_section_ids, token_section_counts = enumerate_section_ids_lmc(
        ids, section_pos_idxs, token_vocab, section_vocab)

    token_section_samples = {}
    for k, (sids, sp) in token_section_counts.items():
        rand_sids = np.random.choice(sids, size=100, replace=True, p=sp)
        start_idx = 0
        token_section_samples[k] = [start_idx, rand_sids]

    token_vocab.truncate()

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Training on {}...'.format(device_str))

    batcher = SkipGramBatchLoader(len(ids), section_pos_idxs, batch_size=args.batch_size)

    model = LMC(args, token_vocab.size(), section_vocab.size()).to(args.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if args.restore_experiment is not None:
        prev_args, model, token_vocab, section_vocab, optimizer_state = restore_model(args.restore_experiment)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    if args.restore_experiment is not None:
        optimizer.load_state_dict(optimizer_state)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)

    batch_incr = batcher.sample_next if args.sample_sections else batcher.marginal_next
    token_section_data = token_section_samples if args.sample_sections else token_section_counts

    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        batcher.reset()
        num_batches = batcher.num_batches()
        epoch_loss = 0.0
        for i in tqdm(range(num_batches)):
            # Reset gradients
            optimizer.zero_grad()

            batch_ids, batch_p = batch_incr(
                ids, full_section_ids, token_section_data, token_vocab, args.window,
                max_num_sections=section_vocab.size()
            )
            batch_ids = list(map(lambda x: torch.LongTensor(x).to(args.device), batch_ids))
            batch_p = (list(map(lambda x: torch.FloatTensor(x).to(args.device), batch_p)) if batch_p[0] is not None else
                       list(batch_p))

            loss = model(*(batch_ids + batch_p))
            if len(loss.size()) > 0:
                loss = loss.mean(0)
            loss.backward()  # backpropagate loss

            epoch_loss += loss.item()
            optimizer.step()
        epoch_loss /= float(batcher.num_batches())
        sleep(0.1)
        print('Epoch={}. Loss={}.'.format(epoch, epoch_loss))
        assert not batcher.has_next()

        # Serializing everything from model weights and optimizer state, to to loss function and arguments
        losses_dict = {'losses': {'kl_loss': epoch_loss}}
        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        save_checkpoint(
            args, model, optimizer, token_vocab, section_vocab, losses_dict, token_section_counts, checkpoint_fp=checkpoint_fp)
