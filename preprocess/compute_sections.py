from collections import defaultdict
import os
import re

import argparse
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--mimic_fp', default='data/mimic/NOTEEVENTS')
    arguments.add_argument('-debug', default=False, action='store_true')

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    debug_str = '_mini' if args.debug else ''
    df = pd.read_csv('{}{}.csv'.format(args.mimic_fp, debug_str))

    sections = defaultdict(int)
    texts = df['TEXT'].tolist()

    for i in tqdm(range(len(texts))):
        pattern = "^([A-Z][A-z0-9 ]+[A-z])(:)"
        matches = re.findall(pattern, texts[i], re.M)

        for match in matches:
            sections[match[0]] += 1

    df_arr = []
    for section in sections:
        count = sections[section]
        df_arr.append((section, sections[section]))
    df = pd.DataFrame(df_arr, columns=['section', 'count'])
    df.to_csv('data/mimic/sections.csv', index=False)
