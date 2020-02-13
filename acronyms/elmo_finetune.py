from collections import defaultdict
import os
from shutil import rmtree
import sys
from time import sleep

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

os.environ['BIDIRECTIONAL_LM_VOCAB_PATH'] = '/home/ga2530/ClinicalBayesianSkipGram/eval/baselines/elmo/'
os.environ['BIDIRECTIONAL_LM_TRAIN_PATH'] = '/home/ga2530/ClinicalBayesianSkipGram/eval/'

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/eval/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/acronyms/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from acronym_utils import get_pretrained_elmo
from elmo_acronym_expander import ELMoAcronymExpander
from error_analysis import elmo_analyze
from eval_utils import lf_tokenizer, preprocess_minnesota_dataset
from fine_tune import load_casi, load_mimic
from model_utils import tensor_to_np


def run_test_epoch(model, test_batcher, indexer, vocab, sf_tokenized_lf_map, loss_func):
    model.eval()  # just sets .requires_grad = True
    test_batcher.reset(shuffle=False)
    test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
    for _ in tqdm(range(test_batcher.num_batches())):
        batch_input, num_outputs = test_batcher.elmo_next(vocab, indexer, sf_tokenized_lf_map)
        batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0).to('cuda'), batch_input))
        with torch.no_grad():
            scores, target = model(*batch_input + [num_outputs])
        num_correct = len(np.where(tensor_to_np(torch.argmax(scores, 1)) == tensor_to_np(target))[0])
        num_examples = len(num_outputs)
        batch_loss = loss_func.forward(scores, target)

        test_correct += num_correct
        test_examples += num_examples
        test_epoch_loss += batch_loss.item()
    sleep(0.1)
    test_loss = test_epoch_loss / float(test_batcher.num_batches())
    test_acc = test_correct / float(test_examples)
    print('Test Loss={}. Accuracy={}'.format(test_loss, test_acc))
    sleep(0.1)
    return test_loss


def elmo_finetune(args, loader):
    args.metadata = None
    train_batcher, test_batcher, train_df, test_df, used_sf_lf_map = loader(args)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('../acronyms', 'weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)
    results_dir = os.path.join('../acronyms', weights_dir, 'results')
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, 'confusion'))

    elmo = get_pretrained_elmo()
    model = ELMoAcronymExpander(elmo).to('cuda')
    indexer = ELMoTokenCharactersIndexer()
    vocab = elmo._lm.vocab

    sf_tokenized_lf_map = defaultdict(list)
    for sf, lf_list in used_sf_lf_map.items():
        for lf in lf_list:
            tokens = lf_tokenizer(lf)
            sf_tokenized_lf_map[sf].append(tokens)

    # Instantiate Adam optimizer
    trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    print('{} trainable params'.format(len(trainable_params)))
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    best_weights = None
    best_epoch = 1
    lowest_test_loss = run_test_epoch(model, test_batcher, indexer, vocab, sf_tokenized_lf_map, loss_func)
    elmo_analyze(test_batcher, model, used_sf_lf_map, vocab, sf_tokenized_lf_map, indexer, results_dir=results_dir)

    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))

        model.train()  # just sets .requires_grad = True
        train_batcher.reset(shuffle=True)
        train_epoch_loss, train_examples, train_correct = 0.0, 0, 0
        for _ in tqdm(range(train_batcher.num_batches())):
            optimizer.zero_grad()

            batch_input, num_outputs = train_batcher.elmo_next(vocab, indexer, sf_tokenized_lf_map)
            batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0).to('cuda'), batch_input))
            scores, target = model(*batch_input + [num_outputs])
            num_correct = len(np.where(tensor_to_np(torch.argmax(scores, 1)) == tensor_to_np(target))[0])
            num_examples = len(num_outputs)
            batch_loss = loss_func.forward(scores, target)

            batch_loss.backward()
            optimizer.step()

            train_correct += num_correct
            train_examples += num_examples
            train_epoch_loss += batch_loss.item()

        sleep(0.1)
        train_loss = train_epoch_loss / float(train_batcher.num_batches())
        train_acc = train_correct / float(train_examples)
        print('Train Loss={}. Accuracy={}'.format(train_loss, train_acc))
        sleep(0.1)

        test_loss = run_test_epoch(model, test_batcher, indexer, vocab, sf_tokenized_lf_map, loss_func)
        lowest_test_loss = min(lowest_test_loss, test_loss)
        if lowest_test_loss == test_loss:
            best_weights = model.state_dict()
            best_epoch = epoch

    print('Loading weights from {} epoch to perform error analysis'.format(best_epoch))
    model.load_state_dict(best_weights)
    elmo_analyze(test_batcher, model, used_sf_lf_map, vocab, sf_tokenized_lf_map, indexer, results_dir=results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Acronym Training Model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--experiment', default='elmo', help='Save path in weights/ for experiment.')
    parser.add_argument('--dataset', default='casi', help='casi or mimic')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--window', default=10, type=int)
    args = parser.parse_args()

    # Load Data
    data_dir = '../eval/eval_data/minnesota/'
    data_fp = os.path.join(data_dir, 'preprocessed_dataset_window_{}.csv'.format(10))
    if not os.path.exists(data_fp):
        print('Need to preprocess dataset first...')
        preprocess_minnesota_dataset(window=10)
        print('Saving dataset to {}'.format(data_fp))

    args.experiment += '_{}'.format(args.dataset)
    loader = load_casi if args.dataset.lower() == 'casi' else load_mimic
    elmo_finetune(args, loader)
