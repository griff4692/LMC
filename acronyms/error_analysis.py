import pandas as pd

from collections import defaultdict

import argparse
from pycm import *
from pylab import axes, boxplot, savefig, title
import numpy as np
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/acronyms/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from acronym_utils import process_batch
from model_utils import tensor_to_np


def _render_example(sf, target_lf, converted_target_lf, pred_lf, top_pred_lfs, context_window, full_context):
    str = 'SF={}.\nTarget LF={} ({}).\nPredicted LF={}.\nTop 5 Predicted={}\n'.format(
        sf, target_lf, converted_target_lf, pred_lf, top_pred_lfs)
    str += 'Context Window: {}\n'.format(context_window)
    str += 'Full Context: {}\n'.format(full_context)
    str += '\n\n'
    return str


def analyze(args, test_batcher, model, used_sf_lf_map, loss_func, vocab, sf_tokenized_lf_map, results_dir=None):
    """
    :param args: argparse.ArgumentParser instance
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
    correct_vars, error_vars = [], []
    for _ in tqdm(range(test_batcher.num_batches())):
        with torch.no_grad():
            batch_loss, num_examples, num_correct, proba, var = process_batch(
                test_batcher, model, loss_func, vocab, sf_tokenized_lf_map)
        test_set_examples += num_examples
        test_set_correct += num_correct
        batch_data = test_batcher.get_prev_batch()
        proba = tensor_to_np(proba)
        var = tensor_to_np(var)
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
                error_vars.append(float(var[batch_idx]))
            else:
                correct_vars.append(float(var[batch_idx]))
            sf_confusion[sf][0].append(target_lf_idx)
            sf_confusion[sf][1].append(pred_lf_idx)

    print('Correct Var Mean={}. Error Var Mean={}'.format(sum(correct_vars) / float(len(correct_vars)),
                                                          sum(error_vars) / float(len(error_vars))))
    with open(os.path.join(results_dir, 'variances.json'), 'w') as fd:
        json.dump({'error': error_vars, 'correct': correct_vars}, fd)
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

    ax = axes()
    boxplot([correct_vars, error_vars])
    title('Distribution of Posterior Variances')
    ax.set_xticklabels(['Correct', 'Error'])
    savefig('{}.png'.format(os.path.join(results_dir, 'variances')))

    pd.DataFrame(df, columns=cols).to_csv(summary_fp, index=False)
    with open(reports_fp, 'w') as fd:
        for report in reports:
            fd.write(report)
    _analyze(args)


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


def _analyze(args):
    # Load Data
    data_dir = os.path.join('../acronyms/weights', args.experiment, 'results')
    summary = pd.read_csv(os.path.join(data_dir, 'summary.csv'))

    suffixes = ['precision', 'recall', 'f1']
    types = ['weighted', 'macro']
    for t in types:
        for suffix in suffixes:
            key = '{}_{}'.format(t, suffix)
            avg_val = summary[key].mean()
            print('Global {} --> {}'.format(key, avg_val))

    num_targets = summary['num_targets'].unique().tolist()
    print('Num Targets, Macro F1, Weighted F1')
    for t in sorted(num_targets):
        avg_macro_f1 = summary[summary['num_targets'] == t]['macro_f1'].mean()
        avg_weighted_f1 = summary[summary['num_targets'] == t]['weighted_f1'].mean()
        print('{},{},{}'.format(t, avg_macro_f1, avg_weighted_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Acronym Fine Tuning Model')

    # Functional Arguments
    parser.add_argument('--experiment', default='submission-baseline')

    args = parser.parse_args()
    analyze(args)
