from collections import defaultdict
import os
from shutil import rmtree
import sys
from time import sleep


import argparse
import numpy as np
import pandas as pd
from scipy.stats import describe
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'acronyms', 'modules'))
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from acronym_utils import lf_tokenizer
from bert_acronym_expander import BERTAcronymExpander
from error_analysis import bert_analyze
from evaluate import load_casi, load_columbia, load_mimic
from model_utils import tensor_to_np


def bert_evaluate(args, loader, train_frac=0.0):
    args.metadata = None
    train_batcher, test_batcher, train_df, test_df, used_sf_lf_map = loader(
        args, batch_size=args.batch_size, train_frac=train_frac)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join(home_dir, 'weights', 'acronyms', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)
    results_dir = os.path.join(weights_dir, 'results')
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, 'confusion'))

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = BERTAcronymExpander(bert_model)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_str)

    sf_tokenized_lf_map = defaultdict(list)
    for sf, lf_list in used_sf_lf_map.items():
        for lf in lf_list:
            tokens = lf_tokenizer(lf)
            sf_tokenized_lf_map[sf].append(tokens)

    return bert_analyze(
        test_batcher, model, used_sf_lf_map, tokenizer, sf_tokenized_lf_map, results_dir=results_dir)


def run_test_epoch(model, test_batcher, indexer, vocab, sf_tokenized_lf_map, loss_func):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()  # just sets .requires_grad = True
    test_batcher.reset(shuffle=False)
    test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
    for _ in tqdm(range(test_batcher.num_batches())):
        batch_input, num_outputs = test_batcher.elmo_next(vocab, indexer, sf_tokenized_lf_map)
        batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0).to(device_str), batch_input))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Acronym Evaluation on Pre-Trained ClinicalBERT')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--experiment', default='bert', help='Save path in weights/ for experiment.')
    parser.add_argument('--dataset', default='casi', help='casi or mimic')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('-bootstrap', default=False, action='store_true')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--window', default=10, type=int)
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

    cols = ['accuracy', 'weighted_f1', 'macro_f1', 'log_loss']
    if args.bootstrap:
        train_frac = 0.2
        iters = 100
    else:
        iters = 1
        train_frac = 0.0

    agg_metrics = []
    for i in range(iters):
        metrics = bert_evaluate(args, dataset_loader, train_frac=train_frac)
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
