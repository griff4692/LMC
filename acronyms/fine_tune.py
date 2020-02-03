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
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/lmc_context/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from acronym_batcher import AcronymBatcherLoader
from acronym_expander import AcronymExpander
from acronym_expander_lmc import AcronymExpanderLMC
from acronym_utils import process_batch
from bsg_utils import restore_model as restore_bsg, save_checkpoint as save_bsg
from error_analysis import analyze, render_test_statistics
from eval_utils import lf_tokenizer, preprocess_minnesota_dataset
from lmc_context_utils import save_checkpoint as lmc_context_save, restore_model as lmc_context_restore
from mimic_tokenize import create_document_token
from model_utils import get_git_revision_hash, render_args


def run_test_epoch(args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map,
                   sf_lf_map, token_metadata_counts, results_dir=None):
    test_batcher.reset(shuffle=False)
    test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
    model.eval()
    for _ in tqdm(range(test_batcher.num_batches())):
        with torch.no_grad():
            batch_loss, num_examples, num_correct, _, _ = process_batch(
                args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_lf_map, sf_tokenized_lf_map,
                token_metadata_counts)
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

    analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map,
            token_metadata_counts, results_dir=results_dir)

    return test_loss


def run_train_epoch(args, train_batcher, model, loss_func, optimizer, token_vocab, metadata_vocab, sf_tokenized_lf_map,
                    sf_lf_map, token_metadata_counts):
    train_batcher.reset(shuffle=True)
    train_epoch_loss, train_examples, train_correct = 0.0, 0, 0
    model.train()
    for _ in tqdm(range(train_batcher.num_batches())):
        optimizer.zero_grad()
        batch_loss, num_examples, num_correct, _, _ = process_batch(
            args, train_batcher, model, loss_func, token_vocab, metadata_vocab, sf_lf_map, sf_tokenized_lf_map,
            token_metadata_counts)
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
    data_fp = os.path.join(data_dir, 'preprocessed_dataset_window_{}.csv'.format(prev_args.window))
    if not os.path.exists(data_fp):
        print('Need to preprocess dataset first...')
        preprocess_minnesota_dataset(window=prev_args.window, combine_phrases=prev_args.combine_phrases)
        print('Saving dataset to {}'.format(data_fp))
    df = pd.read_csv(data_fp)
    df['tokenized_context_unique'] = df['tokenized_context'].apply(lambda x: list(set(x.split())))
    df['row_idx'] = list(range(df.shape[0]))

    with open(os.path.join(data_dir, 'sf_lf_map.json'), 'r') as fd:
        sf_lf_map = json.load(fd)

    train_df, test_df = train_test_split(df, random_state=1992, test_size=0.2)
    train_batcher = AcronymBatcherLoader(train_df, batch_size=32)
    test_batcher = AcronymBatcherLoader(df, batch_size=128)
    # assert len(set(train_df['row_idx'].tolist()).intersection(set(test_df['row_idx'].tolist()))) == 0
    return train_batcher, test_batcher, train_df, test_df, sf_lf_map


def load_mimic(prev_args):
    with open('../eval/eval_data/minnesota/sf_lf_map.json', 'r') as fd:
        sf_lf_map = json.load(fd)
    used_sf_lf_map = {}
    window = 10
    df = pd.read_csv('../context_extraction/data/mimic_rs_dataset_preprocessed_window_{}.csv'.format(window))
    df['tokenized_context_unique'] = df['tokenized_context'].apply(lambda x: list(set(x.split())))
    df['category'] = df['category'].apply(create_document_token)
    if prev_args.metadata == 'category':
        df['metadata'] = df['category']
    else:
        df['metadata'] = df['section']
        df['metadata'].fillna('<pad>', inplace=True)
    sfs = df['sf'].unique().tolist()
    for sf in sfs:
        used_sf_lf_map[sf] = sf_lf_map[sf]
    train_df = df[df['is_train']]
    test_df = df
    train_batcher = AcronymBatcherLoader(train_df, batch_size=32)
    test_batcher = AcronymBatcherLoader(test_df, batch_size=128)
    # assert len(set(train_df['row_idx'].tolist()).intersection(set(test_df['row_idx'].tolist()))) == 0
    return train_batcher, test_batcher, train_df, test_df, used_sf_lf_map


def acronyms_finetune(args, acronym_model, loader, restore_func, save_func):
    args.git_hash = get_git_revision_hash()
    render_args(args)

    if args.lm_type == 'bsg':
        prev_args, lm, token_vocab, _ = restore_func(args.lm_experiment)
        metadata_vocab = None
        prev_args.metadata = None
    else:
        prev_args, lm, token_vocab, metadata_vocab, _, _ = restore_func(args.lm_experiment)
    train_batcher, test_batcher, train_df, test_df, sf_lf_map = loader(prev_args)
    args.metadata = prev_args.metadata

    lf_metadata_counts = None
    if args.metadata is not None:
        metadata_file = '../context_extraction/data/{}_marginals.json'.format(args.metadata)
        with open(metadata_file, 'r') as fd:
            lf_metadata_counts = json.load(fd)
            for sf, lfs in sf_lf_map.items():
                for lf in lfs:
                    if lf not in lf_metadata_counts:
                        raise Exception('No metadata counts for {}'.format(lf))

        for lf, counts in lf_metadata_counts.items():
            names = counts[args.metadata]
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
                args.metadata: trunc_names
            }

    canonical_lfs = pd.read_csv('../eval/eval_data/minnesota/labeled_sf_lf_map.csv')
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

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('../acronyms', 'weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)
    results_dir = os.path.join('../acronyms', weights_dir, 'results')
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, 'confusion'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = acronym_model(args, lm, token_vocab, metadata_vocab).to(device)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    loss_func = nn.CrossEntropyLoss()
    best_weights = model.state_dict()
    best_epoch = 1
    lowest_test_loss = run_test_epoch(args, test_batcher, model, loss_func, token_vocab, metadata_vocab,
                                      sf_tokenized_lf_map, sf_lf_map, lf_metadata_counts,
                                      results_dir=results_dir)

    losses_dict = {
        'train': None,
        'test_loss': lowest_test_loss
    }

    # Make sure it's calculating gradients
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        train_loss = run_train_epoch(args, train_batcher, model, loss_func, optimizer, token_vocab, metadata_vocab,
                                     sf_tokenized_lf_map, sf_lf_map, lf_metadata_counts)
        test_loss = run_test_epoch(
            args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map, sf_lf_map,
            lf_metadata_counts, results_dir=results_dir)

        losses_dict = {
            'train': train_loss,
            'test_loss': test_loss
        }

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
    save_func(args, model, optimizer, token_vocab, losses_dict,
              checkpoint_fp=checkpoint_fp, metadata_vocab=metadata_vocab)
    analyze(args, test_batcher, model, sf_lf_map, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map,
            lf_metadata_counts, results_dir=results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Acronym Training Model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--experiment', default='submission-baseline', help='Save path in weights/ for experiment.')
    parser.add_argument('--lm_experiment', default='baseline-1-13')
    parser.add_argument('--lm_type', default='bsg')
    parser.add_argument('--dataset', default='casi', help='casi or mimic')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-random_encoder', default=False, action='store_true', help='Don\'t use pretrained encoder')
    parser.add_argument('-random_decoder', default=False, action='store_true', help='Don\'t use pretrained decoder')

    args = parser.parse_args()
    args.experiment += '_{}'.format(args.dataset)
    loader = load_casi if args.dataset.lower() == 'casi' else load_mimic
    restore_func = restore_bsg if args.lm_type == 'bsg' else lmc_context_restore
    save_func = save_bsg if args.lm_type == 'bsg' else lmc_context_save
    acronym_model = AcronymExpander if args.lm_type == 'bsg' else AcronymExpanderLMC
    acronyms_finetune(args, acronym_model, loader, restore_func, save_func)
