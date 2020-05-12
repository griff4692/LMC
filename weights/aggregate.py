import os

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Utility script to extract best performing evaluation metrics.')
    parser.add_argument('--experiments')
    parser.add_argument('--lm_type', default='bsg')
    args = parser.parse_args()

    print(args.experiments)
    experiments = args.experiments.split(',')
    datasets = ['mimic', 'casi']
    stat_names = ['log_loss', 'accuracy', 'weighted_f1', 'macro_f1']

    result_cols = []
    results_df = []

    for experiment in experiments:
        df = pd.read_csv(os.path.join(args.lm_type, experiment, 'metrics_5.csv'))
        df['joint_loss'] = df['lm_kl'] + df['lm_recon']
        best = df[df['joint_loss'] == df['joint_loss'].min()]
        for dataset in datasets:
            dataset_vals = best[best['dataset'] == dataset].iloc[0].to_dict()
            stats = list(map(lambda sn: dataset_vals[sn], stat_names))
            results_df.append([experiment, dataset] + stats)

    results_df = pd.DataFrame(results_df, columns=['experiment', 'dataset'] + stat_names)
    for dataset in datasets:
        dataset_df = results_df[results_df['dataset'] == dataset]
        print('Dataset={}'.format(dataset))
        worst_model = dataset_df[dataset_df['log_loss'] == dataset_df['log_loss'].max()]
        best_model = dataset_df[dataset_df['log_loss'] == dataset_df['log_loss'].min()]
        for stat in stat_names:
            vals = dataset_df[stat]
            print('\t{} --> Worst={}, Avg={}, Best={}'.format(
                stat, round(worst_model[stat].tolist()[0], 3), round(vals.mean(), 3), round(best_model[stat].tolist()[0], 3)))
