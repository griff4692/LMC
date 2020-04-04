import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('context_extraction/data/mimic_rs_preprocessed.csv')

    sfs = df['sf'].unique().tolist()

    for sf in sfs:
        sf_df = df[df['sf'] == sf]
        print('SF={}'.format(sf))
        lfs = sf_df['target_lf_sense'].unique().tolist()
        for lf in lfs:
            lf_df = sf_df[sf_df['target_lf_sense'] == lf]
            cat_counts = dict(lf_df['section'].value_counts())
            print('\t', lf, '->', cat_counts)
        print('\n\n')
