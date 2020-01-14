import json
import re

import pandas as pd

UMLS_BLACKLIST =['unidentified', 'otherwise', 'specified', 'nos', 'procedure', 'in abo system',
                 'geographic location']


def clean(lf, sf):
    str = lf.lower()
    tokens = str.split(';')
    token_bag = set()
    strip_regex = r'\s+[(]?({})[)]?'.format('|'.join(UMLS_BLACKLIST))
    for token in tokens:
        token_clean = re.sub(strip_regex, '', token)
        token_clean = re.sub(r'{} - '.format(sf), '', token_clean)
        if not token_clean == sf:
            token_bag.add(token_clean)
    tokens_cleaned = ';'.join(list(sorted(list(token_bag))))
    return tokens_cleaned


if __name__ == '__main__':
    with open('data/sf_lf_map.json', 'r') as fd:
        sf_lf_map = json.load(fd)
    data = []

    for sf, lfs in sf_lf_map.items():
        for lf in lfs:
            data.append((
                sf, lf, clean(lf, sf.lower())
            ))

    df = pd.DataFrame(data, columns=['sf', 'orig_lf', 'lf'])
    df.to_csv('data/lfs.csv', index=False)
