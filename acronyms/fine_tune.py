from collections import defaultdict
from shutil import rmtree
from time import sleep

import argparse
import numpy as np
import pandas as pd
from pycm import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/acronyms/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from acronym_batcher import AcronymBatcherLoader
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


def _render_example(sf, target_lf, converted_target_lf, pred_lf, top_pred_lfs, context_window, full_context):
    str = 'SF={}.\nTarget LF={} ({}).\nPredicted LF={}.\nTop 5 Predicted={}\n'.format(
        sf, target_lf, converted_target_lf, pred_lf, top_pred_lfs)
    str += 'Context Window: {}\n'.format(context_window)
    str += 'Full Context: {}\n'.format(full_context)
    str += '\n\n'
    return str


def error_analysis(test_batcher, model, used_sf_lf_map, loss_func, vocab, sf_tokenized_lf_map, results_dir=None):
    """
    :param test_batcher: AcronymBatcherLoader instance
    :param model: AcronymExpander instance
    :param used_sf_lf_map: Short form to LF mappings
    :param loss_func: PyTorch CrossEntropyLoss instance
    :param vocab: Vocab instance storing tokens and corresponding token ids
    :param results_dir: where to write the results files
    :return: None but writes a confusion matrix analysis file into results_dir
    """
    test_batcher.reset(shuffle=False)
    model.eval()
    sf_confusion = defaultdict(lambda: ([], []))

    results_str = defaultdict(str)
    errors_str = defaultdict(str)
    k = 5
    test_set_examples, test_set_correct = 0, 0
    for _ in tqdm(range(test_batcher.num_batches())):
        with torch.no_grad():
            batch_loss, num_examples, num_correct, proba = process_batch(
                test_batcher, model, loss_func, vocab, sf_tokenized_lf_map)
        test_set_examples += num_examples
        test_set_correct += num_correct
        batch_data = test_batcher.get_prev_batch()
        proba = tensor_to_np(proba)
        top_pred_idxs = np.argsort(-proba, axis=1)[:, :k]
        pred_lf_idxs = top_pred_idxs[:, 0]
        for batch_idx, (row_idx, row) in enumerate(batch_data.iterrows()):
            row = row.to_dict()
            sf = row['sf']
            lf_map = used_sf_lf_map[sf]
            target_lf = row['target_lf']
            target_lf_idx = row['used_target_lf_idx']
            pred_lf_idx = pred_lf_idxs[batch_idx]
            pred_lf = lf_map[pred_lf_idx]
            top_pred_lfs = ', '.join(
                list(map(lambda lf: lf_map[lf], top_pred_idxs[batch_idx][:min(k, len(lf_map))])))
            example_str = _render_example(sf, target_lf, lf_map[target_lf_idx], pred_lf, top_pred_lfs,
                                          row['trimmed_tokens'], row['tokenized_context'])

            results_str[sf] += example_str
            if not target_lf_idx == pred_lf_idx:
                errors_str[sf] += example_str
            sf_confusion[sf][0].append(target_lf_idx)
            sf_confusion[sf][1].append(pred_lf_idx)

    print('Micro Accuracy: {}'.format(test_set_correct / float(test_set_examples)))
    results_fp = os.path.join(results_dir, 'results.txt')
    reports_fp = os.path.join(results_dir, 'reports.txt')
    errors_fp = os.path.join(results_dir, 'errors.txt')
    summary_fp = os.path.join(results_dir, 'summary.csv')
    df = defaultdict(list)
    cols = [
        'sf',
        'support',
        'num_targets',
        'micro_precision',
        'micro_recall',
        'micro_f1',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'weighted_precision',
        'weighted_recall',
        'weighted_f1',
    ]
    reports = []
    with open(results_fp, 'w') as fd:
        for k in sorted(results_str.keys()):
            fd.write(results_str[k])
    with open(errors_fp, 'w') as fd:
        for k in sorted(errors_str.keys()):
            fd.write(errors_str[k])
    for sf in sf_confusion:
        labels = used_sf_lf_map[sf]
        labels_trunc = list(map(lambda x: x.split(';')[0], labels))
        y_true = sf_confusion[sf][0]
        y_pred = sf_confusion[sf][1]
        sf_results = classification_report(y_true, y_pred, labels=list(range(len(labels_trunc))),
                                           target_names=labels_trunc, output_dict=True)
        report = classification_report(y_true, y_pred, labels=list(range(len(labels_trunc))),
                                       target_names=labels_trunc)

        macro_nonzero = defaultdict(float)
        num_nonzero = 0
        for lf in labels_trunc:
            d = sf_results[lf]
            if d['support'] > 0:
                macro_nonzero['precision'] += d['precision']
                macro_nonzero['recall'] += d['recall']
                macro_nonzero['f1-score'] += d['f1-score']
                num_nonzero += 1

        for suffix in ['precision', 'recall', 'f1-score']:
            sf_results['macro avg'][suffix] = macro_nonzero[suffix] / float(num_nonzero)
        reports.append(report)
        reports.append('\n\n')
        metrics = ['micro avg', 'macro avg', 'weighted avg']
        for metric in metrics:
            if metric in sf_results:
                for k, v in sf_results[metric].items():
                    if not k == 'support':
                        metric_key = '{}_{}'.format(metric.split(' ')[0], k.split('-')[0])
                        df[metric_key].append(v)
            else:
                for suffix in ['precision', 'recall', 'f1']:
                    df['{}_{}'.format(metric.split(' ')[0], suffix)].append(None)
        df['sf'].append(sf)
        df['num_targets'].append(len(labels_trunc))
        df['support'].append(sf_results['weighted avg']['support'])
        try:
            cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
            label_idx_to_str = dict()
            for idx in cm.classes:
                label_idx_to_str[idx] = labels_trunc[int(idx)]
            cm.relabel(mapping=label_idx_to_str)
            cm_outpath = os.path.join(results_dir, 'confusion', sf)
            cm.save_html(cm_outpath)
        except:
            print('Only 1 target class for test set SF={}'.format(sf))

    pd.DataFrame(df, columns=cols).to_csv(summary_fp, index=False)
    with open(reports_fp, 'w') as fd:
        for report in reports:
            fd.write(report)


def render_test_statistics(df, sf_lf_map):
    N = df.shape[0]
    sfs = df['sf'].unique().tolist()
    sf_counts = df['sf'].value_counts()

    dominant_counts = 0
    expected_random_accuracy = 0.0
    for sf in sfs:
        dominant_counts += df[df['sf'] == sf]['target_lf_idx'].value_counts().max()
        expected_random_accuracy += sf_counts[sf] / float(len(sf_lf_map[sf]))

    print('Accuracy from choosing dominant class={}'.format(dominant_counts / float(N)))
    print('Expected random accuracy={}'.format(expected_random_accuracy / float(N)))


def process_batch(batcher, model, loss_func, vocab, sf_tokenized_lf_map):
    batch_input, num_outputs = batcher.next(vocab, sf_tokenized_lf_map)
    batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0), batch_input))
    proba, target = model(*(batch_input + [num_outputs]))
    num_correct = len(np.where(tensor_to_np(torch.argmax(proba, 1)) == tensor_to_np(target))[0])
    num_examples = len(num_outputs)
    batch_loss = loss_func.forward(proba, target)
    return batch_loss, num_examples, num_correct, proba


def run_test_epoch(args, test_batcher, model, loss_func, vocab, sf_tokenized_lf_map):
    test_batcher.reset(shuffle=False)
    test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
    model.eval()
    for _ in tqdm(range(test_batcher.num_batches())):
        with torch.no_grad():
            batch_loss, num_examples, num_correct, _ = process_batch(
                test_batcher, model, loss_func, vocab, sf_tokenized_lf_map)
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
    return test_loss


def run_train_epoch(args, train_batcher, model, loss_func, optimizer, vocab, sf_tokenized_lf_map):
    train_batcher.reset(shuffle=True)
    train_epoch_loss, train_examples, train_correct = 0.0, 0, 0
    for _ in tqdm(range(train_batcher.num_batches())):
        optimizer.zero_grad()
        batch_loss, num_examples, num_correct, _ = process_batch(
            train_batcher, model, loss_func, vocab, sf_tokenized_lf_map)
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


def acronyms_finetune(args):
    args.git_hash = get_git_revision_hash()
    render_args(args)

    prev_args, bsg_model, vocab, _ = restore_model(args.bsg_experiment)

    # Load Data
    data_dir = '../eval/eval_data/minnesota/'
    sense_fp = os.path.join(data_dir, 'sense_inventory_ii')
    lfs, lf_sf_map, sf_lf_map = parse_sense_df(sense_fp)
    df = pd.read_csv(os.path.join(data_dir, 'preprocessed_dataset_window_{}.csv'.format(prev_args.window)))
    df['target_lf_idx'] = df['sf'].combine(df['target_lf'], lambda sf, lf: target_lf_index(lf, sf_lf_map[sf]))
    prev_N = df.shape[0]
    df = df[df['target_lf_idx'] > -1]
    print('Removed {} examples for which the target LF is not exactly in the sense inventory ii'.format(
        prev_N - df.shape[0]))

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

    sf_tokenized_lf_map = {}
    for sf, lf_list in used_sf_lf_map.items():
        sf_tokenized_lf_map[sf] = list(map(lf_tokenizer, lf_list))

    train_df, test_df = train_test_split(df, random_state=1992, test_size=0.2)
    train_batcher = AcronymBatcherLoader(train_df, batch_size=args.batch_size)
    test_batcher = AcronymBatcherLoader(test_df, batch_size=args.batch_size)

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

    model = AcronymExpander(bsg_model)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    loss_func = nn.CrossEntropyLoss()
    best_weights = None
    best_epoch = 1
    lowest_test_loss = run_test_epoch(args, test_batcher, model, loss_func, vocab, sf_tokenized_lf_map)

    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))

        train_loss = run_train_epoch(args, train_batcher, model, loss_func, optimizer, vocab, sf_tokenized_lf_map)
        test_loss = run_test_epoch(args, test_batcher, model, loss_func, vocab, sf_tokenized_lf_map)

        losses_dict = {
            'train': train_loss,
            'test_loss': test_loss
        }

        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        save_checkpoint(args, model, optimizer, vocab, losses_dict, checkpoint_fp=checkpoint_fp)

        lowest_test_loss = min(lowest_test_loss, test_loss)
        best_weights = model.state_dict()
        if lowest_test_loss == test_loss:
            best_epoch = epoch

        if args.debug:
            break
    print('Loading weights from {} epoch to perform error analysis'.format(best_epoch))
    model.load_state_dict(best_weights)
    losses_dict['test_loss'] = lowest_test_loss
    checkpoint_fp = os.path.join(weights_dir, 'checkpoint_best.pth')
    save_checkpoint(args, model, optimizer, vocab, losses_dict, checkpoint_fp=checkpoint_fp)
    error_analysis(test_batcher, model, used_sf_lf_map, loss_func, vocab, sf_tokenized_lf_map, results_dir=results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Acronym Training Model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--experiment', default='submission-baseline', help='Save path in weights/ for experiment.')
    parser.add_argument('--bsg_experiment', default='baseline-12-16')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    args = parser.parse_args()
    acronyms_finetune(args)
