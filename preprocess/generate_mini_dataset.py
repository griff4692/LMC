import pandas as pd


if __name__ == '__main__':
    """
    Simple script that takes MIMIC-III NOTEEVENTS.csv file and samples 100 rows / documents.
    This will be the input when you preprocess with the -debug flag.
    """
    df = pd.read_csv('data/mimic/NOTEEVENTS.csv')
    MINI_NUM_DOCS = 100
    df_mini = df.sample(n=MINI_NUM_DOCS, replace=False, random_state=1992)
    mini_out_fn = 'data/mimic/NOTEEVENTS_mini.csv'
    print('Saving mini MIMIC-III dataset with {} documents for development purposes to {}'.format(
        MINI_NUM_DOCS, mini_out_fn))
    df_mini.to_csv(mini_out_fn, index=False)
