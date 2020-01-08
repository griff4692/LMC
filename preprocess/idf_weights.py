from collections import defaultdict
import json

import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    with open('data/mimic/NOTEEVENTS_tokenized.json', 'r') as fd:
        data = json.load(fd)
    doc_freq = defaultdict(int)
    for i in tqdm(range(len(data))):
        tokens = set(data[i][1].split())
        for token in tokens:
            doc_freq[token] += 1

    df = pd.DataFrame(doc_freq.items(), columns=['token', 'df'])
    df.to_csv('data/token_df.csv', index=False)
