from collections import defaultdict
from shutil import rmtree
from time import sleep

import argparse
import pandas as pd
from pycm import *
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/acronyms/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/eval/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from acronym_batcher import AcronymBatcherLoader
from acronym_expander import AcronymExpander
from acronym_expansion import parse_sense_df
from acronym_utils import process_batch, target_lf_index
from bsg_utils import restore_model, save_checkpoint
from error_analysis import analyze, render_test_statistics
from eval_utils import lf_tokenizer, preprocess_minnesota_dataset
from model_utils import get_git_revision_hash, render_args


def run_test_epoch(args, test_batcher, model, loss_func, vocab, sf_tokenized_lf_map, used_sf_lf_map, results_dir=None):
    test_batcher.reset(shuffle=False)
    test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
    model.eval()
    for _ in tqdm(range(test_batcher.num_batches())):
        with torch.no_grad():
            batch_loss, num_examples, num_correct, _, top_global_weights = process_batch(
                args, test_batcher, model, loss_func, vocab, sf_tokenized_lf_map)
        test_correct += num_correct
        test_examples += num_examples
        test_epoch_loss += batch_loss.item()
        if args.debug:
            break
    sleep(0.1)
    test_loss = test_epoch_loss / float(test_batcher.num_batches())
    test_acc = test_correct / float(test_examples)
    print('Test Loss={}. Accuracy={}'.format(test_loss, test_acc))
    sleep(0.1)

    analyze(args, test_batcher, model, used_sf_lf_map, loss_func, vocab, sf_tokenized_lf_map, results_dir=results_dir)

    return test_loss


def run_train_epoch(args, train_batcher, model, loss_func, optimizer, vocab, sf_tokenized_lf_map):
    train_batcher.reset(shuffle=True)
    train_epoch_loss, train_examples, train_correct = 0.0, 0, 0
    for _ in tqdm(range(train_batcher.num_batches())):
        optimizer.zero_grad()
        batch_loss, num_examples, num_correct, _, _ = process_batch(
            args, train_batcher, model, loss_func, vocab, sf_tokenized_lf_map)
        batch_loss.backward()
        optimizer.step()

        # Update metrics
        train_epoch_loss += batch_loss.item()
        train_examples += num_examples
        train_correct += num_correct
        if args.debug:
            break

    sleep(0.1)
    train_loss = train_epoch_loss / float(train_batcher.num_batches())
    train_acc = train_correct / float(train_examples)
    print('Train Loss={}. Accuracy={}'.format(train_loss, train_acc))
    sleep(0.1)
    return train_loss


def load_casi(prev_args):
    # Load Data
    data_dir = '../eval/eval_data/minnesota/'
    sense_fp = os.path.join(data_dir, 'sense_inventory_ii')
    lfs, lf_sf_map, sf_lf_map = parse_sense_df(sense_fp)
    data_fp = os.path.join(data_dir, 'preprocessed_dataset_window_{}.csv'.format(prev_args.window))
    if not os.path.exists(data_fp):
        print('Need to preprocess dataset first...')
        preprocess_minnesota_dataset(window=prev_args.window, combine_phrases=prev_args.combine_phrases)
        print('Saving dataset to {}'.format(data_fp))
    df = pd.read_csv(data_fp)
    df['target_lf_idx'] = df['sf'].combine(df['target_lf'], lambda sf, lf: target_lf_index(lf, sf_lf_map[sf]))
    prev_N = df.shape[0]
    df = df[df['target_lf_idx'] > -1]
    print('Removed {} examples for which the target LF is not exactly in the sense inventory ii'.format(
        prev_N - df.shape[0]))
    df['tokenized_context_unique'] = df['tokenized_context'].apply(lambda x: list(set(x.split())))

    prev_N = df.shape[0]
    df.drop_duplicates(subset=['target_lf', 'tokenized_context'], inplace=True)
    N = df.shape[0]
    print('Removed {} examples with duplicate context-target LF pairs'.format(prev_N - N))

    sfs = df['sf'].unique().tolist()
    used_sf_lf_map = defaultdict(list)
    dominant_sfs = set()

    for sf in sfs:
        subset_df = df[df['sf'] == sf]
        used_target_idxs = subset_df['target_lf_idx'].unique()
        if len(used_target_idxs) == 1:
            dominant_sfs.add(sf)
        else:
            for lf_idx in used_target_idxs:
                used_sf_lf_map[sf].append(sf_lf_map[sf][lf_idx])

    prev_N = df.shape[0]
    df = df[~df['sf'].isin(dominant_sfs)]
    print(('Removing {} examples from {} SF\'s because they have only 1 sense associated with'
           ' them after preprocessing'.format(prev_N - df.shape[0], len(dominant_sfs))))

    df['used_target_lf_idx'] = df['sf'].combine(df['target_lf'], lambda sf, lf: target_lf_index(lf, used_sf_lf_map[sf]))

    df['row_idx'] = list(range(df.shape[0]))
    train_df, test_df = train_test_split(df, random_state=1992, test_size=0.2)
    train_batcher = AcronymBatcherLoader(train_df, batch_size=32)
    test_batcher = AcronymBatcherLoader(test_df, batch_size=32)
    assert len(set(train_df['row_idx'].tolist()).intersection(set(test_df['row_idx'].tolist()))) == 0
    return train_batcher, test_batcher, train_df, test_df, used_sf_lf_map, sfs


def load_mimic(prev_args):
    with open('../context_extraction/data/sf_lf_map.json', 'r') as fd:
        sf_lf_map = json.load(fd)
    used_sf_lf_map = {}
    df = pd.read_csv('../context_extraction/data/mimic_rs_dataset_preprocessed_window_{}.csv'.format(prev_args.window))
    df['tokenized_context_unique'] = df['tokenized_context'].apply(lambda x: list(set(x.split())))
    sfs = df['sf'].unique().tolist()
    for sf in sfs:
        used_sf_lf_map[sf] = sf_lf_map[sf]
    train_df = df[df['is_train']]
    test_df = df[~df['is_train']]
    train_batcher = AcronymBatcherLoader(train_df, batch_size=32)
    test_batcher = AcronymBatcherLoader(test_df, batch_size=32)
    assert len(set(train_df['row_idx'].tolist()).intersection(set(test_df['row_idx'].tolist()))) == 0
    assert len(set(train_df['tokenized_context'].tolist()).intersection(
        set(test_df['tokenized_context'].tolist()))) == 0
    return train_batcher, test_batcher, train_df, test_df, used_sf_lf_map, sfs


def acronyms_finetune(args, loader):
    args.git_hash = get_git_revision_hash()
    render_args(args)

    prev_args, bsg_model, vocab, _ = restore_model(args.bsg_experiment)
    train_batcher, test_batcher, train_df, test_df, used_sf_lf_map, sfs = loader(prev_args)

    sf_tokenized_lf_map = defaultdict(list)
    prev_vocab_size = vocab.size()
    for sf, lf_list in used_sf_lf_map.items():
        for lf in lf_list:
            tokens = lf_tokenizer(lf, vocab)
            sf_tokenized_lf_map[sf].append(tokens)
            for t in tokens:
                vocab.add_token(t)

    for t in sfs:
        vocab.add_token(t.lower())
    new_vocab_size = vocab.size()
    print('Added {} tokens to vocabulary from LF targets and SFs.'.format(new_vocab_size - prev_vocab_size))

    render_test_statistics(test_df, used_sf_lf_map)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('../acronyms', 'weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)
    results_dir = os.path.join('../acronyms', weights_dir, 'results')
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, 'confusion'))

    model = AcronymExpander(args, bsg_model, vocab)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    loss_func = nn.CrossEntropyLoss()
    best_weights = model.state_dict()
    best_epoch = 1
    lowest_test_loss = run_test_epoch(
        args, test_batcher, model, loss_func, vocab, sf_tokenized_lf_map, used_sf_lf_map, results_dir=results_dir)

    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))

        train_loss = run_train_epoch(args, train_batcher, model, loss_func, optimizer, vocab, sf_tokenized_lf_map)
        test_loss = run_test_epoch(
            args, test_batcher, model, loss_func, vocab, sf_tokenized_lf_map, used_sf_lf_map, results_dir=results_dir)

        losses_dict = {
            'train': train_loss,
            'test_loss': test_loss
        }

        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        save_checkpoint(args, model, optimizer, vocab, losses_dict, checkpoint_fp=checkpoint_fp)

        lowest_test_loss = min(lowest_test_loss, test_loss)
        if lowest_test_loss == test_loss:
            best_weights = model.state_dict()
            best_epoch = epoch

        if args.debug:
            break
    print('Loading weights from {} epoch to perform error analysis'.format(best_epoch))
    model.load_state_dict(best_weights)
    losses_dict['test_loss'] = lowest_test_loss
    checkpoint_fp = os.path.join(weights_dir, 'checkpoint_best.pth')
    save_checkpoint(args, model, optimizer, vocab, losses_dict, checkpoint_fp=checkpoint_fp)
    analyze(args, test_batcher, model, used_sf_lf_map, loss_func, vocab, sf_tokenized_lf_map, results_dir=results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Acronym Training Model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--experiment', default='submission-baseline', help='Save path in weights/ for experiment.')
    parser.add_argument('--bsg_experiment', default='baseline-1-13')
    parser.add_argument('--dataset', default='casi', help='casi or mimic')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-random_encoder', default=False, action='store_true', help='Don\'t use pretrained encoder')
    parser.add_argument('-random_priors', default=False, action='store_true', help='Don\'t use pretrained priors')
    parser.add_argument('-use_att', default=False, action='store_true')
    parser.add_argument('--att_style', default='weighted', help='weighted or two_step')

    args = parser.parse_args()
    args.experiment += '_{}'.format(args.dataset)
    loader = load_casi if args.dataset.lower() == 'casi' else load_mimic
    acronyms_finetune(args, loader)
