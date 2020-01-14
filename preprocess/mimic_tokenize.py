from collections import defaultdict
import json
from multiprocessing import Pool
import os
import re
import string
import sys
from time import time

import argparse
from nltk.corpus import stopwords
import pandas as pd
import spacy

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from model_utils import render_args

section_df = pd.read_csv('../preprocess/data/mimic/section.csv').dropna()
SECTION_NAMES = list(sorted(section_df.nlargest(100, columns=['count'])['section'].tolist()))
SECTION_NAMES_LOWER = list(set(list(map(lambda x: x.lower(), SECTION_NAMES))))
SECTION_REGEX = r'\b({})\b:'.format('|'.join(SECTION_NAMES_LOWER))

nlp = spacy.load('en_core_sci_sm')

OTHER_NO = set(['\'s', '`'])
USE_PHRASES = False
swords = set(stopwords.words('english')).union(
    set(string.punctuation)).union(OTHER_NO) - set(['%', '+', '-', '>', '<', '='])
with open('../preprocess/data/prepositions.txt', 'r') as fd:
    prepositions = set(map(lambda x: x.strip(), fd.readlines()))
STOPWORDS = swords - prepositions


def create_section_token(section):
    section = re.sub('[:\s]+', '', section).upper()
    return 'header={}'.format(section)


def create_document_token(category):
    category = re.sub('[:\s]+', '', category).upper()
    return 'document={}'.format(category)


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


def preprocess_mimic(input):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize words
    4. lowercase
    """
    category, text = input
    tokenized_text = []
    sectioned_text = list(filter(lambda x: len(x) > 0 and not x == ' ', re.split(SECTION_REGEX, text)))
    for tok_idx, toks in enumerate(sectioned_text):
        if len(toks) == 0:
            continue
        elif toks in SECTION_NAMES_LOWER:
            if tok_idx + 1 == len(sectioned_text) or not sectioned_text[tok_idx + 1] in SECTION_NAMES_LOWER:
                tokenized_text += [create_section_token(toks)]
        else:
            if args.split_sentences:
                for sentence in nlp(toks).sents:
                    tokens = tokenize_str(str(sentence))
                    if len(tokens) > 1:
                        tokenized_text += [create_section_token('SENTENCE')] + tokens
            else:
                tokenized_text += tokenize_str(toks)
    doc_boundary = [create_document_token(category)]
    return ' '.join(doc_boundary + tokenized_text)


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--mimic_fp', default='data/mimic/NOTEEVENTS')
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('-combine_phrases', default=False, action='store_true')
    arguments.add_argument('-split_sentences', default=False, action='store_true')

    args = arguments.parse_args()
    render_args(args)
    USE_PHRASES = args.combine_phrases
    if USE_PHRASES:
        print('Combining ngrams into phrases')

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''
    sentence_str = '_sentence' if args.split_sentences else ''
    df = pd.read_csv('{}{}{}.csv'.format(args.mimic_fp, '_clean', debug_str))

    print('Loaded {} rows of data. Tokenizing...'.format(df.shape[0]))
    categories = df['CATEGORY'].tolist()
    text = df['TEXT'].tolist()
    start_time = time()
    p = Pool()
    parsed_docs = p.map(preprocess_mimic, zip(categories, text))
    p.close()
    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))

    token_cts = defaultdict(int)
    for doc_idx, doc in enumerate(parsed_docs):
        for token in doc.split():
            token_cts[token] += 1
            token_cts['__ALL__'] += 1
    debug_str = '_mini' if args.debug else ''
    with open(args.mimic_fp + '_tokenized{}{}{}.json'.format(debug_str, phrase_str, sentence_str), 'w') as fd:
        json.dump(parsed_docs, fd)
    with open(args.mimic_fp + '_token_counts{}{}{}.json'.format(debug_str, phrase_str, sentence_str), 'w') as fd:
        json.dump(token_cts, fd)
