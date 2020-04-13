import os

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Utility script to split datasets for evaluation metrics.')
    parser.add_argument('--experiment')
    parser.add_argument('--lm_type', default='bsg')
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.lm_type, args.experiment, 'metrics.csv'))
    cols = ['accuracy', 'macro_f1', 'weighted_f1']
    datasets = ['mimic', 'casi', 'columbia']
    for dataset in datasets:
        sub_df = df[df['dataset'] == dataset][cols]
        sub_df.to_csv(os.path.join(args.lm_type, args.experiment, 'metrics_{}.csv'.format(dataset)), index=False)
