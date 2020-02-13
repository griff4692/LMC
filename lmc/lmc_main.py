import os
from shutil import rmtree
import sys
from time import sleep

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from lmc_model import LMC
from lmc_prebatch import precompute
from lmc_utils import restore_model, save_checkpoint
from model_utils import get_git_revision_hash, render_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-multi_gpu', default=False, action='store_true')
    parser.add_argument('--window', default=10, type=int)

    # Model Hyperparameters
    parser.add_argument('--metadata_samples', default=5, type=int)
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

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Training on {}...'.format(device_str))

    cached_file = '../preprocess/data/batches{}.pth'.format(debug_str)
    if not os.path.exists(cached_file):
        precompute(args)
    batch_data = torch.load(cached_file)
    token_vocab = batch_data['token_vocab']
    metadata_vocab = batch_data['metadata_vocab']
    data_loader = batch_data['data_loader']
    token_metadata_counts = batch_data['token_metadata_counts']

    if args.restore:
        _, model, token_vocab, metadata_vocab, optimizer_state, token_metadata_counts = restore_model(args.experiment)
    else:
        model = LMC(args, token_vocab.size(), metadata_vocab.size()).to(args.device)
        optimizer_state = None

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        args.batch_size *= torch.cuda.device_count()

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
        num_batches = len(data_loader)
        epoch_joint_loss, epoch_kl_loss, epoch_recon_loss = 0.0, 0.0, 0.0
        for i, batch_ids in tqdm(enumerate(data_loader), total=num_batches):
            # Reset gradients
            optimizer.zero_grad()
            batch_ids = list(map(lambda x: x.to(args.device), batch_ids))
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

        epoch_joint_loss /= float(num_batches)
        epoch_kl_loss /= float(num_batches)
        epoch_recon_loss /= float(num_batches)
        sleep(0.1)
        print('Epoch={}. Joint loss={}.  KL Loss={}. Reconstruction Loss={}'.format(
            epoch, epoch_joint_loss, epoch_kl_loss, epoch_recon_loss))

        # Serializing everything from model weights and optimizer state, to to loss function and arguments
        losses_dict = {'losses': {'joint': epoch_joint_loss, 'kl': epoch_kl_loss, 'recon': epoch_recon_loss}}
        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        if epoch < 10:
            save_checkpoint(args, model, optimizer, token_vocab, losses_dict, token_metadata_counts,
                            checkpoint_fp=checkpoint_fp, metadata_vocab=metadata_vocab)
