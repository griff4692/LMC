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

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from model_utils import render_args
from compute_sections import HEADER_SEARCH_REGEX

# Loaded so it's available inside scope of preprocess MIMIC without re-loading for every document or having to pickle
section_df = pd.read_csv(os.path.join(home_dir, 'preprocess/data/mimic/section_freq.csv')).dropna()
SECTION_NAMES = list(set(list(sorted(section_df['section'].tolist()))))
nlp = None


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


def create_section_token(section):
    """
    :param section: string representing a section header as extracted from MIMIC note
    :return: string of format header=SECTIONNAME (i.e. header=HISTORYOFPRESENTILLNESS)
    """
    section = re.sub('[:\s]+', '', section).upper()
    return 'header={}'.format(section)


def create_document_token(category):
    """
    :param section: string representing a note type as provided by MIMIC
    :return: string of format document=DOCUMENTCATEGORY (i.e. document=DISCHARGESUMMARY)
    """
    category = re.sub('[:\s]+', '', category).upper()
    return 'document={}'.format(category)


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


def pattern_repl(matchobj):
    """
    :param matchobj: re.Match object
    :return: Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def preprocess_mimic(input, split_sentences=False):
    """
    :param input: string representing a single MIMIC note
    :param split_sentences: boolean indicating whether to separate sentences with special sentence token
    :return: a string representing space delimited tokenized text
    e.g. document=CONSULT header=DISCHARGEDATE digitparsed header=CHIEFCOMPLAINT back pain

    Extracts section headers with regex but only converts to special header={} token if it's a frequently observed
    section, which is defined as having a corpus count >= 10.
    """
    category, text = input
    stopwords = get_mimic_stopwords()

    tokenized_text = []
    sectioned_text = list(filter(lambda x: len(x.strip()) > 0, re.split(HEADER_SEARCH_REGEX, text, flags=re.M)))
    is_header_arr = list(map(lambda x: re.match(HEADER_SEARCH_REGEX, x, re.M) is not None, sectioned_text))
    for tok_idx, toks in enumerate(sectioned_text):
        is_header = is_header_arr[tok_idx]
        is_next_header = tok_idx + 1 == len(is_header_arr) or is_header_arr[tok_idx + 1]

        if is_header and is_next_header:
            continue
        if is_header:
            header_stripped = toks.strip().strip(':').upper()
            if header_stripped in SECTION_NAMES:
                tokenized_text += [create_section_token(header_stripped)]
            else:
                toks = clean_text(toks)
                tokenized_text += tokenize_str(toks, stopwords=stopwords)
        else:
            if args.split_sentences:
                # for sentence in nlp(toks).sents:
                #     tokens = tokenize_str(str(sentence), stopwords=stopwords)
                #     if len(tokens) > 1:
                #         tokenized_text += [create_section_token('SENTENCE')] + tokens

                sentence_tok = create_section_token('SENTENCE')
                split_toks = list(filter(lambda x: len(x) > 0, re.split(SEP_REGEX, toks)))
                tokenized_text = list(map(
                    lambda toks: ([sentence_tok] if toks[0] > 0 and toks[0] < len(split_toks) - 1 else []) +
                                 tokenize_str(clean_text(toks[1]), stopwords=stopwords), enumerate(split_toks)))
                tokenized_text_flat = list(itertools.chain(*tokenized_text))
                tokenized_text += tokenized_text_flat
            else:
                toks = clean_text(toks)
                tokenized_text += tokenize_str(toks, stopwords=stopwords)
    doc_boundary = [create_document_token(category)]

    return ' '.join(doc_boundary + tokenized_text)


def preprocess_mimic_split_sentences(input):
    return preprocess_mimic(input, split_sentences=True)


def tokenize_str(token_str, stopwords=[]):
    tokens = token_str.lower().strip().split()
    tokens = list(map(lambda x: x.strip(string.punctuation), tokens))
    return list(filter(lambda x: len(x) > 0 and not x == ' ' and x not in stopwords, tokens))


if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC-III Note Tokenization.')
    arguments.add_argument('--mimic_fp', default=os.path.join(home_dir, 'preprocess/data/mimic/NOTEEVENTS'))
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('-split_sentences', default=False, action='store_true')
    arguments.add_argument('-filter_rs', default=False, action='store_true')

    args = arguments.parse_args()
    render_args(args)

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    debug_str = '_mini' if args.debug else ''
    sentence_str = '_sentence' if args.split_sentences else ''
    df = pd.read_csv('{}{}.csv'.format(args.mimic_fp, debug_str))
    N = df.shape[0]

    if args.filter_rs:
        rs_fn = 'context_extraction/data/mimic_rs_dataset.csv'
        doc_ids = []
        if os.path.exists(rs_fn):
            doc_ids = [int(x) for x in pd.read_csv(rs_fn)['doc_id'].unique().tolist()]
        prev_n = N
        df = df[~df['ROW_ID'].isin(doc_ids)]
        N = df.shape[0]
        print('Removed {} documents used in reverse substitution dataset.'.format(prev_n - N))

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
            if 'header=' not in token and 'document=' not in token:
                token_cts['__ALL__'] += 1
    debug_str = '_mini' if args.debug else ''
    out_tok_fn = args.mimic_fp + '_tokenized{}{}.json'.format(debug_str, sentence_str)
    out_counts_fn = args.mimic_fp + '_token_counts{}{}.json'.format(debug_str, sentence_str)
    print('Saving tokens to {} and token counts to {}'.format(out_tok_fn, out_counts_fn))
    with open(out_tok_fn, 'w') as fd:
        json.dump(parsed_docs, fd)
    with open(out_counts_fn, 'w') as fd:
        json.dump(token_cts, fd)
