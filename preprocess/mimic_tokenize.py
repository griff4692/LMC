from collections import defaultdict
import json
from multiprocessing import Pool
import os
import re
import string
from time import time

import argparse
from nltk.corpus import stopwords
import pandas as pd
import spacy

section_df = pd.read_csv('../preprocess/data/mimic/sections.csv').dropna()
SECTION_NAMES = list(sorted(section_df.nlargest(100, columns=['count'])['section'].tolist()))
SECTION_REGEX = r'\b({})\b:'.format('|'.join(SECTION_NAMES))

nlp = spacy.load('en_core_sci_sm')

OTHER_NO = set(['\'s', '`'])
USE_PHRASES = False
STOPWORDS = set(stopwords.words('english')).union(
    set(string.punctuation)).union(OTHER_NO) - set(['%', '+', '-', '>', '<', '='])
with open('../preprocess/data/prepositions.txt', 'r') as fd:
    prepositions = set(map(lambda x: x.strip(), fd.readlines()))
STOPWORDS -= prepositions


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def create_section_token(section):
    section = re.sub('[:\s]+', '', section).upper()
    return 'header={}'.format(section)


def clean_text(text):
    """
    Clean text
    """
    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'[_*?/()]+', ' ', text)
    text = re.sub(r'\b(-)?[\d.]+(-)?\b', ' DIGITPARSED ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def chunk(tokens, chunker=None):
    chunker = chunker or nlp
    token_str = ' '.join(tokens)
    for chunk in chunker(token_str).ents:
        chunk = str(chunk)
        chunk_toks = chunk.strip().split()
        if len(chunk_toks) > 1:
            token_str = token_str.replace(chunk, '_'.join(chunk_toks))
    return token_str.split()


def tokenize_str(token_str, combine_phrases=None, chunker=None):
    tokens = token_str.lower().strip().split()
    tokens = list(map(lambda x: x.strip(string.punctuation), tokens))
    combine_phrases = USE_PHRASES if combine_phrases is None else combine_phrases
    if combine_phrases:
        tokens = chunk(tokens, chunker=chunker)
    return list(filter(lambda x: len(x) > 0 and not x == ' ' and x not in STOPWORDS, tokens))


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
            for sentence in nlp(toks).sents:
                tokens = tokenize_str(str(sentence))
                if len(tokens) > 1:
                    tokenized_text += ([create_section_token('SENTENCESTART')] + tokens +
                                      [create_section_token('SENTENCEEND')])
    doc_boundary = [create_section_token('DOCUMENT')]
    return ' '.join(doc_boundary + tokenized_text)


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--mimic_fp', default='data/mimic/NOTEEVENTS')
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('-combine_phrases', default=False, action='store_true')

    args = arguments.parse_args()
    USE_PHRASES = args.combine_phrases
    if USE_PHRASES:
        print('Combining ngrams into phrases')

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''
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
    with open(args.mimic_fp + '_tokenized{}{}.json'.format(debug_str, phrase_str), 'w') as fd:
        json.dump(list(zip(categories, parsed_docs)), fd)
    with open(args.mimic_fp + '_token_counts{}{}.json'.format(debug_str, phrase_str), 'w') as fd:
        json.dump(token_cts, fd)
