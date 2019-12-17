import os
from shutil import rmtree
import sys
from time import sleep

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from acronyms.batcher import AcronymBatcherLoader
from acronym_expander import AcronymExpander
from acronym_expansion import parse_sense_df
from eval_utils import lf_tokenizer
from model_utils import get_git_revision_hash, render_args, restore_model, save_checkpoint, tensor_to_np


def target_lf_index(target_lf, lfs):
    for i in range(len(lfs)):
        lf_tokens = lfs[i].split(';')
        for lf in lf_tokens:
            if lf.lower() == target_lf.lower():
                return i
    return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Acronym Training Model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../eval/eval_data/minnesota/')
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')
    parser.add_argument('--bsg_experiment', default='baseline-12-16')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    args = parser.parse_args()
    args.git_hash = get_git_revision_hash()
    render_args(args)

    prev_args, bsg_model, vocab, _ = restore_model(args.bsg_experiment, weights_path='../model/weights/')

    # Load Data
    sense_fp = os.path.join(args.data_dir, 'sense_inventory_ii')
    lfs, lf_sf_map, sf_lf_map = parse_sense_df(sense_fp)
    sf_tokenized_lf_map = {}
    for sf, lf_list in sf_lf_map.items():
        sf_tokenized_lf_map[sf] = list(map(lf_tokenizer, lf_list))
    df = pd.read_csv(os.path.join(args.data_dir, 'preprocessed_dataset_window_{}.csv'.format(prev_args.window)))
    df['target_lf_idx'] = df['sf'].combine(df['target_lf'], lambda sf, lf: target_lf_index(lf, sf_lf_map[sf]))
    prev_N = df.shape[0]
    df = df[df['target_lf_idx'] > -1]
    print('Removed {} examples for which the target LF is not exactly in the sense inventory ii'.format(
        prev_N - df.shape[0]))
    train_df, test_df = train_test_split(df, random_state=1992, test_size=0.1)
    train_batcher = AcronymBatcherLoader(train_df, batch_size=args.batch_size)
    test_batcher = AcronymBatcherLoader(test_df, batch_size=args.batch_size)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)

    model = AcronymExpander(bsg_model)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    loss_func = nn.CrossEntropyLoss()

    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        train_batcher.reset(shuffle=True)

        train_epoch_loss, train_examples, train_correct = 0.0, 0, 0
        for _ in tqdm(range(train_batcher.num_batches())):
            optimizer.zero_grad()
            batch_input, num_outputs = train_batcher.next(vocab, sf_tokenized_lf_map)
            batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0), batch_input))

            proba, target = model(*(batch_input + [num_outputs]))
            train_correct += len(np.where(tensor_to_np(torch.argmax(proba, 1)) == tensor_to_np(target))[0])
            train_examples += len(num_outputs)
            batch_loss = loss_func.forward(proba, target)
            batch_loss.backward()
            optimizer.step()

            train_epoch_loss += batch_loss.item()

        sleep(0.1)
        train_loss = train_epoch_loss / float(train_batcher.num_batches())
        train_acc = train_correct / float(train_examples)
        print('Train Loss={}. Accuracy={}'.format(train_loss, train_acc))
        sleep(0.1)
        test_batcher.reset(shuffle=False)
        test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
        model.eval()
        for _ in tqdm(range(test_batcher.num_batches())):
            batch_input, num_outputs = test_batcher.next(vocab, sf_tokenized_lf_map)
            batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0), batch_input))
            with torch.no_grad():
                proba, target = model(*(batch_input + [num_outputs]))
            test_correct += len(np.where(tensor_to_np(torch.argmax(proba, 1)) == tensor_to_np(target))[0])
            test_examples += len(num_outputs)
            batch_loss = loss_func.forward(proba, target)
            test_epoch_loss += batch_loss.item()

        sleep(0.1)
        test_loss = test_epoch_loss / float(test_batcher.num_batches())
        test_acc = test_correct / float(test_examples)
        print('Test Loss={}. Accuracy={}'.format(test_loss, test_acc))
        sleep(0.1)

        losses_dict = {
            'train': train_loss,
            'test_loss': test_loss
        }

        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        save_checkpoint(args, model, optimizer, vocab, losses_dict, checkpoint_fp=checkpoint_fp)
