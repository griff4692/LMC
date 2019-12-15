from collections import defaultdict
import json
from multiprocessing import Pool
import os
import re
import string
from time import time

import argparse
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm

from chunker import chunk

CHUNK = True
OTHER_NO = set(['\'s', '`'])
STOPWORDS = set(stopwords.words('english')).union(
    set(string.punctuation)).union(OTHER_NO) - set(['%', '+', '-', '>', '<', '='])


section_df = pd.read_csv('data/mimic/sections.csv').dropna()
SECTION_NAMES = list(sorted(section_df.nlargest(1000, columns=['count'])['section'].tolist()))
SECTION_REGEX = r'\b({})\b:'.format('|'.join(SECTION_NAMES))


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def create_section_token(section):
    section = re.sub('[:\s]+', '', section)
    return '||header{}||'.format(section)


def clean_text(text):
    """
    Clean text
    """
    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'[_*?]+', ' ', text)
    text = re.sub(r'\b(-)?[\d.]+(-)?\b', ' DIGITPARSED ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize words
    4. lowercase
    """
    cleaned_text = clean_text(text)
    tokenized_text = []
    sectioned_text = list(filter(lambda x: len(x) > 0 and not x == ' ', re.split(SECTION_REGEX, cleaned_text)))
    for tok_idx, toks in enumerate(sectioned_text):
        if len(toks) == 0:
            continue
        elif toks in SECTION_NAMES:
            if tok_idx + 1 == len(sectioned_text) or not sectioned_text[tok_idx + 1] in SECTION_NAMES:
                tokenized_text += [create_section_token(toks)]
        else:
            tokens = [x.strip(string.punctuation) for x in word_tokenize(toks.lower().strip())]
            tokens = filter(lambda x: x not in STOPWORDS, tokens)
            tokenized_text += list(tokens)
    doc_boundary = [create_section_token('DOCUMENT')]
    unigram_text = ' '.join(doc_boundary + tokenized_text)
    if CHUNK:
        return chunk(unigram_text)
    return unigram_text


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

    print('Loaded {} rows of data. Tokenizing...'.format(df.shape[0]))
    categories = df['CATEGORY'].tolist()
    start_time = time()
    p = Pool()
    parsed_docs = p.map(preprocess_mimic, df['TEXT'].tolist())
    p.close()

    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))

    token_cts = defaultdict(int)
    for doc_idx, doc in enumerate(parsed_docs):
        for token in doc.split():
            token_cts[token] += 1
            token_cts['__ALL__'] += 1
    debug_str = '_mini' if args.debug else ''
    chunks_str = '_chunk' if args.chunk else ''
    with open(args.mimic_fp + '_tokenized{}{}.json'.format(debug_str, chunks_str), 'w') as fd:
        json.dump(list(zip(categories, parsed_docs)), fd)
    with open(args.mimic_fp + '_token_counts{}.json'.format(debug_str, chunks_str), 'w') as fd:
        json.dump(token_cts, fd)
