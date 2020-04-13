import os

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Utility script to extract best performing evaluation metrics.')
    parser.add_argument('--experiment')
    parser.add_argument('--lm_type', default='bsg')

    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.lm_type, args.experiment, 'metrics.csv'))

    mimic_df = df[df['dataset'] == 'mimic']
    casi_df = df[df['dataset'] == 'casi']

    lowest_mimic_loss = mimic_df['log_loss'].min()
    optimal_mimic_examples = mimic_df[mimic_df['log_loss'] == lowest_mimic_loss].examples.tolist()[0]

    lowest_casi_loss = casi_df['log_loss'].min()
    optimal_casi_examples = casi_df[casi_df['log_loss'] == lowest_casi_loss].examples.tolist()[0]

    casi_metric_row = casi_df[casi_df['examples'] == optimal_mimic_examples]
    mimic_metric_row = mimic_df[mimic_df['examples'] == optimal_casi_examples]

    casi_dict = casi_metric_row.iloc[0].to_dict()
    mimic_dict = mimic_metric_row.iloc[0].to_dict()

    for d in [casi_dict, mimic_dict]:
        print(d['dataset'])
        for k, v in d.items():
            print('\t{} -> {}'.format(k, v))
        print('\n\n')

    print('Highest MIMIC accuracy={}'.format(mimic_df['accuracy'].max()))
    print('Highest CASI accuracy={}'.format(casi_df['accuracy'].max()))
