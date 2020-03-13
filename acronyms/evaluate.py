from collections import defaultdict
import json
import os
from shutil import rmtree
import sys
from time import sleep

import argparse
import pandas as pd
import torch
import torch.nn as nn

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'acronyms'))
sys.path.insert(0, os.path.join(home_dir, 'acronyms', 'modules'))
sys.path.insert(0, os.path.join(home_dir, 'modules', 'bsg'))
sys.path.insert(0, os.path.join(home_dir, 'modules', 'lmc'))
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from acronym_utils import load_casi, load_mimic, lf_tokenizer, run_test_epoch, run_train_epoch
from bsg_acronym_expander import BSGAcronymExpander
from bsg_utils import restore_model as restore_bsg
from error_analysis import analyze, render_test_statistics
from lmc_acronym_expander import LMCAcronymExpander, LMCBertAcronymExpander
from lmc_prebatch import create_tokenizer_maps
from lmc_utils import restore_model as lmc_restore
from model_utils import get_git_revision_hash, render_args


def extract_smoothed_metadata_probs(metadata='section'):
    """
    :param metadata: What metadata variable to use (i.e. section or category)
    :return: Returns smoothed version of empirical probabilities as computed in preprocess/context_extraction/data/
    """
    if metadata is None:
        return None
    metadata_file = os.path.join(
        home_dir, 'preprocess/context_extraction/data/{}_marginals.json'.format(args.metadata))
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


def run_evaluation(args, dataset_loader, restore_func, train_frac=0.0):
    """
    :param args: argparse instance specifying evaluation configuration (including pre-trained model path)
    :param dataset_loader: function to load acronym expansion dataset (i.e. either CASI or Reverse Substitution MIMIC)
    :param restore_func: Function to load pre-trained model weights (different for BSG and LMC)
    :param train_frac: If you want to fine tune the model, this should be about 0.8.
    Otherwise, default of 0.0 means the entire dataset is used as a test set for evaluation
    :return:
    """
    args.git_hash = get_git_revision_hash()
    render_args(args)

    if args.lm_type == 'bsg':
        prev_args, lm, token_vocab, _ = restore_func(args.lm_experiment)
        metadata_vocab = None
        bert_tokenizer = None
        prev_args.metadata = None
    else:
        prev_args, lm, token_vocab, metadata_vocab, bert_tokenizer, _, _ = restore_func(args.lm_experiment)
    train_batcher, test_batcher, train_df, test_df, sf_lf_map = dataset_loader(prev_args, train_frac=train_frac)
    args.metadata = prev_args.metadata
    args.bert = hasattr(prev_args, 'bert') and prev_args.bert

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

    wp_conversions = None
    if bert_tokenizer is not None:
        additional_tokens = test_df['trimmed_tokens'].tolist() + train_df['trimmed_tokens'].tolist()
        for arr in additional_tokens:
            token_vocab.add_tokens(arr.split(' '))
        wp_conversions = create_tokenizer_maps(bert_tokenizer, token_vocab, metadata_vocab, token_keys=True)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join(home_dir, 'weights', 'acronyms', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)
    results_dir = os.path.join(home_dir, 'acronyms', weights_dir, 'results')
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, 'confusion'))

    if args.lm_type == 'bsg':
        model = BSGAcronymExpander(args, lm, token_vocab).to(args.device)
    elif hasattr(prev_args, 'bert') and prev_args.bert:
        model = LMCBertAcronymExpander(args, lm).to(args.device)
    else:
        model = LMCAcronymExpander(args, lm, token_vocab).to(args.device)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    loss_func = nn.CrossEntropyLoss()
    best_weights = model.state_dict()
    best_epoch = 1
    lowest_test_loss = run_test_epoch(args, test_batcher, model, loss_func, token_vocab, metadata_vocab,
                                      wp_conversions, sf_tokenized_lf_map, sf_lf_map, lf_metadata_counts)
    analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, wp_conversions,
            sf_tokenized_lf_map, lf_metadata_counts, results_dir=results_dir)

    # Make sure it's calculating gradients
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        _ = run_train_epoch(args, train_batcher, model, loss_func, optimizer, token_vocab, metadata_vocab,
                                     wp_conversions, sf_tokenized_lf_map, sf_lf_map, lf_metadata_counts)
        test_loss = run_test_epoch(
            args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map, sf_lf_map,
            lf_metadata_counts)
        analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, wp_conversions,
                sf_tokenized_lf_map, lf_metadata_counts, results_dir=results_dir)

        lowest_test_loss = min(lowest_test_loss, test_loss)
        if lowest_test_loss == test_loss:
            best_weights = model.state_dict()
            best_epoch = epoch
    print('Loading weights from {} epoch to perform error analysis'.format(best_epoch))
    model.load_state_dict(best_weights)
    analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, wp_conversions,
            sf_tokenized_lf_map, lf_metadata_counts, results_dir=results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Clinical Acronym Expansion Evaluation')

    # Functional Arguments
    parser.add_argument('--experiment', default='submission-baseline', help='Save path in weights/ for experiment.')
    parser.add_argument('--lm_experiment', default='baseline-1-13')
    parser.add_argument('--lm_type', default='bsg')
    parser.add_argument('--dataset', default='casi', help='casi or mimic')

    # Training Hyperparameters
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--window', default=10, type=int)

    args = parser.parse_args()
    args.experiment += '_{}'.format(args.dataset)
    dataset_loader = load_casi if args.dataset.lower() == 'casi' else load_mimic
    restore_func = restore_bsg if args.lm_type == 'bsg' else lmc_restore

    args.device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    run_evaluation(args, dataset_loader, restore_func, train_frac=0.0)
