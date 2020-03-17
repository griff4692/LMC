from collections import defaultdict
import itertools
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

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from model_utils import render_args
from compute_sections import HEADER_SEARCH_REGEX
from map_sections import get_canonical_section

nlp = None

SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'
SEP_TOK = '<sep>'
SPECIAL_TOKS = {SEP_TOK}


def remove_phi(text):
    return re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)


def remove_non_eos_periods(text):
    return re.sub(r'(Mr|Dr|Ms|Mrs)\.', r'\1', text)


def clean_text(text):
    """
    :param text: string representing raw MIMIC note
    :return: cleaned string
    - Replace [**Patterns**] with spaces
    - Replace digits with special DIGITPARSED token
    """
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)

    # Replace `_` with spaces.
    text = re.sub(r'[_*?/()]+', ' ', text)
    text = re.sub(r'\b(-)?[\d.,]+(-)?\b', ' DIGITPARSED ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def create_section_token(section, is_main=True):
    """
    :param section: string representing a section header as extracted from MIMIC note
    :return: string of format header=SECTIONNAME (i.e. header=HISTORYOFPRESENTILLNESS)
    """
    section = re.sub('[:\s]+', '', section).upper()
    prefix = 'header' if is_main else 'sub'
    return '{}={}'.format(prefix, section)


def get_mimic_stopwords():
    """
    :return: set containing English stopwords plus non-numeric punctuation.
    Does not include prepositions since these are helpful for detecting nouns
    """
    other_no = set(['\'s', '`'])
    swords = set(stopwords.words('english')).union(
        set(string.punctuation)).union(other_no) - set(['%', '+', '-', '>', '<', '='])
    with open(os.path.join(home_dir, 'shared_data', 'prepositions.txt'), 'r') as fd:
        prepositions = set(map(lambda x: x.strip(), fd.readlines()))
    return swords - prepositions


def is_actual_header(tok):
    return len(tok) > 1 and tok[0].upper() == tok[0] and len(tok) > 0 and tok[-1] == ':'


def pattern_repl(matchobj):
    """
    :param matchobj: re.Match object
    :return: Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def preprocess_columbia(input):
    pass


def preprocess_mimic(input, split_sentences=False):
    """
    :param input: string representing a single MIMIC note
    :param split_sentences: boolean indicating whether to separate sentences with special sentence token
    :return: a string representing space delimited tokenized text
    e.g. header=CONSULT->START header=CONSULT->DISCHARGEDATE digitparsed header=CONSULT->CHIEFCOMPLAINT back pain

    Extracts section headers with regex according to canonical mappings.  Non-standard headings are represented with
    'sub'
    """
    category, text = input
    text = remove_phi(text)
    text = remove_non_eos_periods(text)
    stopwords = get_mimic_stopwords()

    full_tokenized_text = []
    sectioned_text = list(filter(lambda x: len(x.strip()) > 0, re.split(HEADER_SEARCH_REGEX, text, flags=re.M)))
    headers = set(re.findall(HEADER_SEARCH_REGEX, text, flags=re.M))
    is_header_arr = list(map(lambda x: x in headers, sectioned_text))
    for tok_idx, toks in enumerate(sectioned_text):
        is_header = is_header_arr[tok_idx]

        if is_header and is_actual_header(toks):
            header_stripped = toks.strip().strip(':').upper()
            canonical_section = get_canonical_section(header_stripped, category)
            is_canonical = '->' in canonical_section
            if not is_canonical:  # mark this as a context window boundary with <sep> token
                full_tokenized_text.append('<sep>')
            full_tokenized_text.append(create_section_token(canonical_section, is_canonical))
        else:
            if split_sentences:
                for sentence in nlp(toks).sents:
                    tokens = tokenize_str(str(sentence), stopwords=stopwords)
                    if len(tokens) > 1:
                        full_tokenized_text += [create_section_token('SENTENCE')] + tokens
            else:
                split_toks = list(filter(lambda x: len(x) > 0, re.split(SEP_REGEX, toks)))
                tokenized_text = list(map(
                    lambda toks: ([SEP_TOK] if toks[0] > 0 and toks[0] < len(split_toks) - 1 else []) +
                                 tokenize_str(clean_text(toks[1]), stopwords=stopwords), enumerate(split_toks)))
                tokenized_text_flat = list(itertools.chain(*tokenized_text))
                full_tokenized_text += tokenized_text_flat
    doc_boundary = [create_section_token('{}->START'.format(category), True)]
    return ' '.join(doc_boundary + full_tokenized_text)


def preprocess_mimic_split_sentences(input):
    return preprocess_mimic(input, split_sentences=True)


def tokenize_str(token_str, stopwords=[]):
    tokens = token_str.lower().strip().split()
    tokens = list(map(lambda x: x.strip(string.punctuation), tokens))
    return list(filter(lambda x: len(x) > 0 and not x == ' ' and x not in stopwords, tokens))


def load_mimic(args):
    debug_str = '_mini' if args.debug else ''
    remote_fp = '/nlp/corpora/mimic/mimic_iii/NOTEEVENTS.csv'
    if os.path.exists(remote_fp) and not args.debug:
        return pd.read_csv(remote_fp)
    return pd.read_csv('data/mimic/NOTEEVENTS{}.csv'.format(debug_str))


def load_columbia(args):
    year_start = 2014 if args.debug else 1988
    year_range = list(map(str, (range(year_start, 2015))))
    dir = '/nlp/corpora/dsum_corpora/dsum_corpus_02'
    docs = []
    for year in year_range:
        print('Loading year={}'.format(year))
        fn = os.path.join(dir, year)
        with open(fn, 'r') as fd:
            for line in fd:
                doc = re.sub(r'^.+\|\s*', '', line).strip().encode('ascii', errors='ignore').decode()
                docs.append(doc)
    N = len(docs)
    print('Loaded {} documents'.format(N))
    categories = ['Discharge Summary'] * N
    df = pd.DataFrame({'CATEGORY': categories, 'TEXT': docs})
    return df


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC-III/Columbia Discharge Summary Note Tokenization.')
    arguments.add_argument('--dataset', default='mimic')
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('-split_sentences', default=False, action='store_true')

    args = arguments.parse_args()
    render_args(args)

    print('Loading data...')
    if args.dataset == 'mimic':
        df = load_mimic(args)
        processor = preprocess_mimic
        if args.split_sentences:
            nlp = spacy.load('en_core_sci_sm')
            processor = preprocess_mimic_split_sentences
    elif args.dataset == 'columbia':
        df = load_columbia(args)
    else:
        raise Exception('Only supported for Columbia and MIMIC-III')

    sentence_str = '_sentence' if args.split_sentences else ''

    processor = preprocess_mimic
    if args.split_sentences:
        nlp = spacy.load('en_core_sci_sm')
        processor = preprocess_mimic_split_sentences

    print('Loaded {} rows of data. Tokenizing...'.format(df.shape[0]))
    categories = df['CATEGORY'].tolist()
    text = df['TEXT'].tolist()
    start_time = time()
    p = Pool()  # Can specify processes=<digit> for control over how many processes are spawned
    parsed_docs = p.map(processor, zip(categories, text))
    p.close()
    end_time = time()
    print('Took {} seconds'.format(end_time - start_time))

    token_cts = defaultdict(int)
    for doc_idx, doc in enumerate(parsed_docs):
        for token in doc.split():
            token_cts[token] += 1
            # Don't include special tokens in token counts
            if 'header=' not in token and token not in SPECIAL_TOKS:
                token_cts['__ALL__'] += 1
    debug_str = '_mini' if args.debug else ''
    out_fn = 'data/{}/'.format(args.dataset)
    if args.dataset == 'mimic':
        out_fn += 'NOTEEVENTS_'
    out_tok_fn = out_fn + 'tokenized{}{}.json'.format(debug_str, sentence_str)
    out_counts_fn = out_fn + 'token_counts{}{}.json'.format(debug_str, sentence_str)
    print('Saving tokens to {} and token counts to {}'.format(out_tok_fn, out_counts_fn))
    with open(out_tok_fn, 'w') as fd:
        json.dump(parsed_docs, fd)
    with open(out_counts_fn, 'w') as fd:
        json.dump(token_cts, fd)
