from collections import Counter, defaultdict
import json
import os
import random
from shutil import rmtree
import sys
from time import sleep

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from scipy.stats import describe
import torch
import torch.nn as nn

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'acronyms'))
sys.path.insert(0, os.path.join(home_dir, 'acronyms', 'modules'))
sys.path.insert(0, os.path.join(home_dir, 'modules', 'bsg'))
sys.path.insert(0, os.path.join(home_dir, 'modules', 'lmc'))
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from acronym_utils import (load_casi, load_columbia, load_mimic, lf_tokenizer, render_dominant_section_accuracy,
                           split_marginals, run_test_epoch, run_train_epoch)
from bsg_acronym_expander import BSGAcronymExpander
from bsg_utils import restore_model as restore_bsg
from error_analysis import analyze, render_test_statistics
from lmc_acronym_expander import LMCAcronymExpander
from lmc_utils import restore_model as lmc_restore
from model_utils import get_git_revision_hash, render_args


def _generate_marginals(df):
    lfs = df['target_lf_sense'].unique().tolist()

    marginals = {}
    for lf in lfs:
        sections = df[df['target_lf_sense'] == lf]['section_mapped'].tolist()
        section_counts = Counter(sections)
        counts = []
        names = []
        for s, c in section_counts.items():
            names.append(s)
            counts.append(c)

        marginals[lf] = {}
        marginals[lf]['count'] = counts
        tsum = float(sum(counts))
        marginals[lf]['section'] = names
        marginals[lf]['p'] = [c / tsum for c in counts]
    return marginals


def extract_smoothed_metadata_probs(metadata='section'):
    """
    :param metadata: What metadata variable to use (i.e. section or category)
    :return: Returns smoothed version of empirical probabilities as computed in preprocess/context_extraction/data/
    """
    if metadata is None:
        return None
    metadata_file = os.path.join(
        home_dir, 'preprocess/context_extraction/data/{}_marginals.json'.format(metadata))
    with open(metadata_file, 'r') as fd:
        lf_metadata_counts = json.load(fd)

    for lf, counts in lf_metadata_counts.items():
        names = counts[metadata]
        c_arr = counts['count']
        p_arr = counts['p']
        trunc_names = []
        trunc_c = []
        all_ones = max(c_arr) == 1
        for n, c, p in zip(names, c_arr, p_arr):
            if c > 1 or all_ones:
                trunc_names.append(n)
                trunc_c.append(c)

        s = max(float(sum(trunc_c)), 1.0)
        trunc_p = list(map(lambda x: x / s, trunc_c))
        lf_metadata_counts[lf] = {
            'count': trunc_c,
            'p': trunc_p,
            metadata: trunc_names
        }

        return lf_metadata_counts


def run_evaluation(args, acronym_model, dataset_loader, restore_func, train_frac=0.0):
    """
    :param args: argparse instance specifying evaluation configuration (including pre-trained model path)
    :param acronym_model: PyTorch model to rank candidate acronym expansions (an instance of model from ./modules/)
    :param dataset_loader: function to load acronym expansion dataset (i.e. either CASI or Reverse Substitution MIMIC)
    :param restore_func: Function to load pre-trained model weights (different for BSG and LMC)
    :param train_frac: If you want to fine tune the model, this should be about 0.8.
    Otherwise, default of 0.0 means the entire dataset is used as a test set for evaluation
    :return:
    """
    args.git_hash = get_git_revision_hash()
    render_args(args)

    if args.lm_type == 'bsg':
        prev_args, lm, token_vocab, _ = restore_func(args.lm_experiment, ckpt=args.ckpt)
        metadata_vocab = None
        prev_args.metadata = None
    else:
        prev_args, lm, token_vocab, metadata_vocab, _, _, _ = restore_func(args.lm_experiment, ckpt=args.ckpt)
    train_batcher, test_batcher, train_df, test_df, sf_lf_map = dataset_loader(prev_args, train_frac=train_frac)
    args.metadata = prev_args.metadata

    # Construct smoothed empirical probabilities of metadata conditioned on LF ~ p(metadata|LF)
    lf_metadata_counts = extract_smoothed_metadata_probs(metadata=args.metadata)

    casi_dir = os.path.join(home_dir, 'shared_data', 'casi')
    canonical_lfs = pd.read_csv(os.path.join(casi_dir, 'labeled_sf_lf_map.csv'))
    canonical_sf_lf_map = dict(canonical_lfs.groupby('target_lf_sense')['target_label'].apply(list))

    sf_tokenized_lf_map = defaultdict(list)
    prev_vocab_size = token_vocab.size()
    for sf, lf_list in sf_lf_map.items():
        token_vocab.add_token(sf.lower())
        for lf in lf_list:
            canonical_lf_arr = list(set(canonical_sf_lf_map[lf]))
            assert len(canonical_lf_arr) == 1
            canonical_lf = canonical_lf_arr[0]
            tokens = lf_tokenizer(canonical_lf, token_vocab)
            sf_tokenized_lf_map[sf].append(tokens)
            for t in tokens:
                token_vocab.add_token(t)
    new_vocab_size = token_vocab.size()
    print('Added {} tokens to vocabulary from LF targets and SFs.'.format(new_vocab_size - prev_vocab_size))

    render_test_statistics(test_df, sf_lf_map)

    if lf_metadata_counts is not None:
        if args.dataset == 'mimic':
            train_lf_metadata_counts, val_lf_metadata_counts = split_marginals(lf_metadata_counts)
        else:
            train_lf_metadata_counts = lf_metadata_counts
            val_lf_metadata_counts = _generate_marginals(test_df)
        render_dominant_section_accuracy(train_lf_metadata_counts, val_lf_metadata_counts, sf_lf_map)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join(home_dir, 'weights', 'acronyms', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)
    results_dir = os.path.join(home_dir, 'acronyms', weights_dir, 'results')
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, 'confusion'))

    model = acronym_model(args, lm, token_vocab).to(args.device)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    loss_func = nn.CrossEntropyLoss()
    best_weights = model.state_dict()
    best_epoch = 1
    lowest_test_loss, highest_test_acc = run_test_epoch(
        args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map, sf_lf_map,
        lf_metadata_counts)

    metrics = analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map,
                      lf_metadata_counts, results_dir=results_dir)
    metrics['log_loss'] = lowest_test_loss
    metrics['accuracy'] = highest_test_acc
    if args.epochs == 0:
        return metrics

    # Make sure it's calculating gradients
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        _ = run_train_epoch(args, train_batcher, model, loss_func, optimizer, token_vocab, metadata_vocab,
                                     sf_tokenized_lf_map, sf_lf_map, lf_metadata_counts)
        test_loss, test_acc = run_test_epoch(
            args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map, sf_lf_map,
            lf_metadata_counts)
        analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map,
                lf_metadata_counts, results_dir=results_dir)

        lowest_test_loss = min(lowest_test_loss, test_loss)
        highest_test_acc = max(highest_test_acc, test_acc)
        if lowest_test_loss == test_loss:
            best_weights = model.state_dict()
            best_epoch = epoch
    print('Loading weights from {} epoch to perform error analysis'.format(best_epoch))
    model.load_state_dict(best_weights)
    metrics = analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map,
                      lf_metadata_counts, results_dir=results_dir)
    metrics['log_loss'] = lowest_test_loss
    metrics['accuracy'] = highest_test_acc
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Clinical Acronym Expansion Evaluation')

    # Functional Arguments
    parser.add_argument('--experiment', default='submission-baseline', help='Save path in weights/ for experiment.')
    parser.add_argument('--lm_experiment', default='baseline-1-13')
    parser.add_argument('--ckpt', default=None, type=int,
                        help='Optionally preselect an epoch from which to load pretrained.')
    parser.add_argument('--lm_type', default='bsg')
    parser.add_argument('--dataset', default='casi', help='casi or mimic')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--window', default=10, type=int)

    parser.add_argument('-bootstrap', default=False, action='store_true')

    args = parser.parse_args()
    args.experiment += '_{}'.format(args.dataset)
    dl = args.dataset.lower()
    if dl == 'mimic':
        dataset_loader = load_mimic
    elif dl == 'casi':
        dataset_loader = load_casi
    elif dl == 'columbia':
        dataset_loader = load_columbia
    else:
        raise Exception('Didn\'t recognize datset={}'.format(dl))
    restore_func = restore_bsg if args.lm_type == 'bsg' else lmc_restore
    acronym_model = BSGAcronymExpander if args.lm_type == 'bsg' else LMCAcronymExpander
    args.device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    cols = ['accuracy', 'weighted_f1', 'macro_f1', 'log_loss']
    if args.bootstrap:
        train_frac = 0.2
        iters = 100
    else:
        iters = 1
        train_frac = 0.0

    agg_metrics = []
    for i in range(iters):
        metrics = run_evaluation(args, acronym_model, dataset_loader, restore_func, train_frac=train_frac)
        metric_row = [metrics[col] for col in cols]
        agg_metrics.append(metric_row)

    if args.bootstrap:
        df = pd.DataFrame(agg_metrics, columns=cols)
        for col in cols:
            print(col)
            print(describe(df[col].tolist()))
        results_dir = os.path.join(home_dir, 'weights', 'acronyms', args.experiment, 'results')
        bootstrap_fn = os.path.join(results_dir, 'bootstrap.csv')
        print('Saving bootstrap results to {}'.format(bootstrap_fn))
        df.to_csv(bootstrap_fn, index=False)
