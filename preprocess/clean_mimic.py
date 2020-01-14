from multiprocessing import Pool
import os
import re
from time import time

import argparse
import pandas as pd


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def clean_text(text):
    """
    Clean text
    """
    # Replace [**Patterns**] with spaces and lowercase.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'[_*?/()]+', ' ', text)
    text = re.sub(r'\b(-)?[\d.]+(-)?\b', ' DIGITPARSED ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Cleaning.')
    arguments.add_argument('--mimic_fp', default='data/mimic/NOTEEVENTS')
    arguments.add_argument('-debug', default=False, action='store_true')

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    debug_str = '_mini' if args.debug else ''
    df = pd.read_csv('{}{}.csv'.format(args.mimic_fp, debug_str))

    print('Loaded {} rows of data. Cleaning...'.format(df.shape[0]))
    start_time = time()
    p = Pool()
    cleaned_text = p.map(clean_text, df['TEXT'].tolist())
    p.close()
    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))

    df['TEXT'] = cleaned_text
    out_fn = '{}_clean{}.csv'.format(args.mimic_fp, debug_str)
    df.to_csv(out_fn, index=False)
