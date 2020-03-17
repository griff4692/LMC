import numpy as np
import pandas as pd


MAX_NUM = 750  # 10000


if __name__ == '__main__':
    lfs_df = pd.read_csv('data/lfs_w_counts.csv')

    sfs = lfs_df['sf'].unique().tolist()
    lf_to_sf_map = lfs_df.set_index('target_label').to_dict()
    viable_sf_count = 0
    viable_sfs = []
    for sf in sfs:
        subset_df = lfs_df[lfs_df['sf'] == sf]
        counts = subset_df['count'].tolist()
        if min(counts) > 1:
            viable_sf_count += 1
            viable_sfs.append(sf)
    print('SFs with at least 2 contexts for every LF={}'.format(viable_sf_count))
    contexts = pd.read_csv('data/mimic_contexts.csv')
    print('Number of contexts={}'.format(contexts.shape[0]))
    contexts['sf'] = contexts['lf'].apply(lambda lf: lf_to_sf_map['sf'][lf])
    contexts['target_lf_sense'] = contexts['lf'].apply(lambda lf: lf_to_sf_map['target_lf_sense'][lf])
    contexts = contexts[contexts['sf'].isin(viable_sfs)]
    print('Number of contexts after filtering out unviable SFs={}'.format(contexts.shape[0]))
    final_data = []
    for lf in contexts['lf'].unique().tolist():
        lf_df = contexts[contexts['lf'] == lf]
        N = lf_df.shape[0]
        if N > MAX_NUM:
            lf_df = lf_df.sample(n=MAX_NUM, random_state=1992)
            print('Truncating {} from {} to {} examples'.format(lf, N, MAX_NUM))
            N = MAX_NUM

        is_train = np.zeros([N, ])
        indices = np.arange(N)
        num_train = int(0.8 * N)
        if num_train == 0:
            num_train += 1
        elif num_train == N:
            num_train -= 1
        np.random.shuffle(indices)
        train_indices = indices[:num_train]
        is_train[train_indices] = 1
        lf_df['is_train'] = is_train

        final_data.append(lf_df)
    final_df = pd.concat(final_data)
    print('Number of contexts after subsampling={}'.format(final_df.shape[0]))
    final_df['is_train'] = final_df['is_train'].apply(lambda x: x == 1)
    final_df.to_csv('data/mimic_rs_dataset.csv', index=False)
