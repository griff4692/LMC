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
from compute_sections import enumerate_metadata_ids_lmc
from lmc_context_batcher import LMCContextSkipGramBatchLoader
from lmc_context_model import LMCC
from lmc_context_utils import save_checkpoint
from model_utils import get_git_revision_hash, render_args
from vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('-combine_phrases', default=False, action='store_true')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-multi_gpu', default=False, action='store_true')
    parser.add_argument('--window', default=5, type=int)

    # Model Hyperparameters
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension for encoder')
    parser.add_argument('--input_dim', default=100, type=int, help='embedding dimemsions for encoder')
    parser.add_argument('--hinge_loss_margin', default=1.0, type=float, help='reconstruction margin')
    parser.add_argument('--latent_dim', default=100, type=int, help='z dimension')
    parser.add_argument('--metadata', default='section',
                        help='sections or category. What to define latent variable over.')
    parser.add_argument('-same_metadata', default=False, action='store_true')
    parser.add_argument('-sample_metadata', action='store_true', default=False,
                        help='Sample 1 metadata rather than compute full marginal across all metadata.')

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
    print('Loaded vocabulary of size={}...'.format(token_vocab.section_start_vocab_id))

    metadata_counts = pd.read_csv('../preprocess/data/mimic/{}.csv'.format(args.metadata))
    header_prefix = 'header=' if args.metadata == 'section' else 'document='
    metadata_counts['header'] = metadata_counts[args.metadata].apply(lambda x: header_prefix + re.sub(r'\s+', '', x.upper()))
    metadata_counts_dict = metadata_counts.set_index('header').to_dict()['count']

    print('Collecting metadata information for {}...'.format(args.metadata))
    assert token_vocab.section_start_vocab_id <= token_vocab.category_start_vocab_id
    start_id = token_vocab.section_start_vocab_id if args.metadata == 'section' else token_vocab.category_start_vocab_id
    end_id = token_vocab.category_start_vocab_id if args.metadata == 'section' else token_vocab.size()
    metadata_id_range = np.arange(start_id, end_id)
    is_metadata = np.isin(ids, metadata_id_range)
    metadata_pos_idxs = np.where(is_metadata)[0]
    all_metadata_pos_idxs = np.where(ids >= token_vocab.section_start_vocab_id)[0]
    metadata_vocab = Vocab()
    for id in metadata_id_range:
        name = token_vocab.get_token(id)
        metadata_vocab.add_token(name, token_support=int(metadata_counts_dict[name]))
    full_metadata_ids, token_metadata_counts = enumerate_metadata_ids_lmc(
        ids, metadata_pos_idxs, token_vocab, metadata_vocab)

    token_metadata_samples = {}
    for k, (sids, sp) in token_metadata_counts.items():
        size = min(len(sp) * 10, 250)
        rand_sids = np.random.choice(sids, size=size, replace=True, p=sp)
        start_idx = 0
        token_metadata_samples[k] = [start_idx, rand_sids]

    token_vocab.truncate(token_vocab.section_start_vocab_id)
    ids[all_metadata_pos_idxs] = -1

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Training on {}...'.format(device_str))

    model = LMCC(args, token_vocab.size(), metadata_vocab.size()).to(args.device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        args.batch_size *= torch.cuda.device_count()
    batcher = LMCContextSkipGramBatchLoader(len(ids), all_metadata_pos_idxs, batch_size=args.batch_size)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)

    batch_incr = batcher.next_same if args.same_metadata else (
        batcher.sample_next if args.sample_metadata else batcher.marginal_next)
    token_metadata = token_metadata_samples if args.sample_metadata else token_metadata_counts

    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        batcher.reset()
        num_batches = batcher.num_batches()
        epoch_joint_loss, epoch_kl_loss, epoch_recon_loss = 0.0, 0.0, 0.0
        for i in tqdm(range(num_batches)):
            # Reset gradients
            optimizer.zero_grad()

            batch_ids, batch_p = batch_incr(
                ids, full_metadata_ids, token_metadata, token_vocab, args.window,
                max_num_metadata=metadata_vocab.size()
            )
            batch_ids = list(map(lambda x: torch.LongTensor(x).to(args.device), batch_ids))
            batch_p = (list(map(lambda x: torch.FloatTensor(x).to(args.device), batch_p)) if batch_p[0] is not None else
                       list(batch_p))

            kl_loss, recon_loss = model(*(batch_ids + batch_p))
            if len(kl_loss.size()) > 0:
                kl_loss = kl_loss.mean(0)
            if len(recon_loss.size()) > 0:
                recon_loss = recon_loss.mean(0)
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
        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        save_checkpoint(args, model, optimizer, token_vocab, losses_dict, token_metadata_counts,
                        checkpoint_fp=checkpoint_fp, metadata_vocab=metadata_vocab)
