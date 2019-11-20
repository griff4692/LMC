from collections import defaultdict
import json
import os
import re

import argparse
import pandas as pd

from preprocess.BioSentVec.src.preprocess import preprocess_mimic


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--mimic_fp', default='~/Desktop/mimic/NOTEEVENTS')

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    df = pd.read_csv(args.mimic_fp + '.csv', dtype={'TEXT': str})
    print('Loaded data. Tokenizing...')

    categories, parsed_docs = [], []

    MIMIC_BOILERPLATE = {
        'addendum :',
        'admission date :',
        'discharge date :',
        'date of birth :',
        'sex : m',
        'sex : f',
        'service :',
    }

    BOILERPLATE_REGEX = re.compile('|'.join(MIMIC_BOILERPLATE))
    token_cts = defaultdict(int)
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        tokenized_doc = ' '.join(list(preprocess_mimic(row['TEXT'])))
        tokenized_cleaned_doc = re.sub(BOILERPLATE_REGEX, '', tokenized_doc).strip()
        tokenized_cleaned_doc = re.sub('\s+', ' ', tokenized_cleaned_doc)
        for token in tokenized_cleaned_doc.split():
            token_cts[token] += 1
            token_cts['__ALL__'] += 1
        parsed_docs.append(tokenized_cleaned_doc)
        categories.append(row['CATEGORY'])
        if row_idx + 1 % 100 == 0:
            print('Processed {} out of {} rows'.format(row_idx + 1, df.shape[0]))
    json.dump(list(zip(categories, parsed_docs)), open(args.mimic_fp + '_tokenized.json', 'w'))
    json.dump(token_cts, open(args.mimic_fp + '_token_counts.json', 'w'))
