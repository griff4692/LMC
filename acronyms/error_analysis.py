import pandas as pd

from collections import defaultdict
from shutil import rmtree

import argparse
from pycm import *

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/acronyms/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from model_utils import render_args


def analyze(args):
    render_args(args)

    # Load Data
    data_dir = os.path.join('weights', args.experiment, 'results')

    summary = pd.read_csv(os.path.join(data_dir, 'summary.csv'))

    global_macro = defaultdict(float)
    supports = summary['support'].sum()
    N = summary.shape[0]

    suffixes = ['precision', 'recall', 'f1']
    types = ['weighted', 'macro']
    for t in types:
        for suffix in suffixes:
            key = '{}_{}'.format(t, suffix)
            avg_val = summary[key].mean()
            print('Global {} --> {}'.format(key, avg_val))

    num_targets = summary['num_targets'].unique().tolist()
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
