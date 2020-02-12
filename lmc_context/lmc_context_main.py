import pickle
import os
from shutil import rmtree
import sys
from time import sleep

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.insert(0, 'D:/git_codes/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, 'D:/git_codes/ClinicalBayesianSkipGram/utils/')
from compute_sections import enumerate_metadata_ids_lmc
from lmc_context_batcher import LMCContextSkipGramBatchLoader
from lmc_context_model import LMCC
from lmc_context_utils import restore_model, save_checkpoint
from model_utils import get_git_revision_hash, render_args
from vocab import Vocab
from category_vocab import category_Vocab
import csv
from categorize import categorize_token_metadata_counts
sys.path.insert(0, 'D:/git_codes/ClinicalBayesianSkipGram/')

def generate_metadata_samples(token_metadata_counts, metadata_vocab, token_vocab, sample=10):
    token_metadata_samples = {}
    smooth_counts = np.zeros([metadata_vocab.size()])
    all_metadata_ids = np.arange(metadata_vocab.size())
    for k, (sids, sp) in token_metadata_counts.items():
        size = [min(len(sp) * 100, 10000), sample]
        # Add smoothing
        smooth_counts.fill(1.0)
        smooth_counts[sids] += sp
        smooth_p = smooth_counts / smooth_counts.sum()
        rand_sids = np.random.choice(all_metadata_ids, size=size, replace=True, p=smooth_p)
        start_idx = 0
        token_metadata_samples[k] = [start_idx, rand_sids]

    token_vocab.truncate(token_vocab.section_start_vocab_id)
    return token_metadata_samples

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default=sys.path[0])
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('-combine_phrases', default=False, action='store_true')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-multi_gpu', default=False, action='store_true')
    parser.add_argument('--window', default=10, type=int)

    # Model Hyperparameters
    parser.add_argument('--metadata_samples', default=10, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension for encoder')
    parser.add_argument('--input_dim', default=100, type=int, help='embedding dimemsions for encoder')
    parser.add_argument('--hinge_loss_margin', default=1.0, type=float, help='reconstruction margin')
    parser.add_argument('--metadata', default='section',
                        help='sections or category. What to define latent variable over.')
    parser.add_argument('--recon_coeff', default=1.0, type=float)
    parser.add_argument('-restore', default=False, action='store_true')

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

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Training on {}...'.format(device_str))

    # Load Vocabulary
    vocab_infile = os.path.join(args.data_dir, 'vocab{}{}.pk'.format(debug_str, phrase_str))
    print('Loading vocabulary from {}...'.format(vocab_infile))
    with open(vocab_infile, 'rb') as fd:
        token_vocab = pickle.load(fd)
    print('Loaded vocabulary of size={}...'.format(token_vocab.section_start_vocab_id))

    print('Collecting metadata information for {}...'.format(args.metadata))
    assert token_vocab.section_start_vocab_id <= token_vocab.category_start_vocab_id
    start_id = token_vocab.category_start_vocab_id
    end_id = token_vocab.size()
    metadata_id_range = np.arange(start_id, end_id)
    is_metadata = np.isin(ids, metadata_id_range)
    metadata_pos_idxs = np.where(is_metadata)[0]
    all_metadata_pos_idxs = np.where(ids >= token_vocab.section_start_vocab_id)[0]
    metadata_vocab = Vocab()
    for id in metadata_id_range:
        name = token_vocab.get_token(id)
        metadata_vocab.add_token(name)
    full_metadata_ids, token_metadata_counts = enumerate_metadata_ids_lmc(
        ids, metadata_pos_idxs, token_vocab, metadata_vocab)
    ids[all_metadata_pos_idxs] = -1

    categories_path = sys.path[0] + 'CATEGORIES.csv'
    with open(categories_path, "r") as f:
        reader = csv.reader(f)
        categories_list = list(reader)
        
    category_vocab = category_Vocab()
    category_vocab.store_categories(categories_list)
    category_dict = dict()
    for i in range(len(categories_list)):
        category_dict['document={}'.format(i)] = [category_vocab.get_ids(x) for x in categories_list[i]]
    print("Tokenized {} number of distinct categories".format(category_vocab.size()))
    
    categorize_token_metadata_counts(token_metadata_counts,metadata_vocab,category_dict);

    if args.restore:
        _, model, token_vocab, metadata_vocab, optimizer_state, token_metadata_counts = restore_model(args.experiment)
        token_metadata_samples = generate_metadata_samples(
            token_metadata_counts, category_vocab, token_vocab, sample=args.metadata_samples)
    else:
        token_metadata_samples = generate_metadata_samples(
            token_metadata_counts, category_vocab, token_vocab, sample=args.metadata_samples)
        model = LMCC(args, token_vocab.size(), category_vocab.size()).to(args.device)
        optimizer_state = None

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        args.batch_size *= torch.cuda.device_count()
    batcher = LMCContextSkipGramBatchLoader(len(ids), all_metadata_pos_idxs, batch_size=args.batch_size)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('weights', args.experiment)
    if os.path.exists(weights_dir) and not args.restore:
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    
    
    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    epoch = 3 if args.restore else 1
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        batcher.reset()
        num_batches = batcher.num_batches()
        epoch_joint_loss, epoch_kl_loss, epoch_recon_loss = 0.0, 0.0, 0.0
        for i in tqdm(range(num_batches)):
            # Reset gradients
            optimizer.zero_grad()

            batch_ids = batcher.next(ids, full_metadata_ids, token_metadata_samples, token_vocab, args.window)
            batch_ids = list(map(lambda x: torch.LongTensor(x).to(args.device), batch_ids))
            kl_loss, recon_loss = model(*batch_ids)
            if len(kl_loss.size()) > 0:
                kl_loss = kl_loss.mean(0)
            if len(recon_loss.size()) > 0:
                recon_loss = recon_loss.mean(0)
            joint_loss = kl_loss + recon_loss
            joint_adj_loss = kl_loss + args.recon_coeff * recon_loss
            joint_loss.backward()  # backpropagate loss
            optimizer.step()

            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_joint_loss += joint_loss.item()

            if (i + 1) % 10000 == 0:
                print('Saving Checkpoint at Batch={}'.format(i + 1))
                d = float(i + 1)
                # Serializing everything from model weights and optimizer state, to to loss function and arguments
                losses_dict = {'losses': {'joint': epoch_joint_loss / d,
                                          'kl': epoch_kl_loss / d,
                                          'recon': epoch_recon_loss / d}
                               }
                print(losses_dict)
                checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
                if epoch < 10:
                    save_checkpoint(args, model, optimizer, token_vocab, losses_dict, token_metadata_counts,
                                    checkpoint_fp=checkpoint_fp, metadata_vocab=metadata_vocab)

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
        if epoch < 10:
            save_checkpoint(args, model, optimizer, token_vocab, losses_dict, token_metadata_counts,
                            checkpoint_fp=checkpoint_fp, metadata_vocab=metadata_vocab)
