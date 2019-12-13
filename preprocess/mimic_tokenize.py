from collections import defaultdict
import json
from multiprocessing import Pool
import os
import re

import argparse
from nltk import word_tokenize
import pandas as pd
from tqdm import tqdm


"""
Preprocess MIMIC-III reports
"""

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

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text


def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        yield ' '.join(word_tokenize(sec)).lower()


def test_preprocess_mimic():
    text = """Normal sinus rhythm. Right bundle-branch block with secondary ST-T wave
    abnormalities. Compared to the previous tracing of [**2198-5-26**] no diagnostic
    interim change.

    """
    sents = [sen for sen in preprocess_mimic(text)]
    assert sents[0] == 'normal sinus rhythm .'
    assert sents[1] == 'right bundle-branch block with secondary st-t wave abnormalities .'
    assert sents[2] == 'compared to the previous tracing of no diagnostic interim change .'


def preprocess_mimic_doc(text):
    tokenized_doc = ' '.join(list(preprocess_mimic(text)))
    tokenized_cleaned_doc = re.sub(BOILERPLATE_REGEX, '', tokenized_doc).strip()
    tokenized_cleaned_doc = re.sub('\s+', ' ', tokenized_cleaned_doc)
    return tokenized_cleaned_doc


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
    
    p = Pool(processes=10)
    parsed_docs = tqdm(p.map(preprocess_mimic_doc, df['TEXT'].tolist()))
    p.close()

    print('Finished tokenization.  Now collecting word counts.')

    token_cts = defaultdict(int)
    for doc in parsed_docs:
        for token in doc.split():
            token_cts[token] += 1
            token_cts['__ALL__'] += 1
    debug_str = '_mini' if args.debug else ''
    with open(args.mimic_fp + '_tokenized{}.json'.format(debug_str), 'w') as fd:
        json.dump(list(zip(categories, parsed_docs)), fd)
    with open(args.mimic_fp + '_token_counts{}.json'.format(debug_str), 'w') as fd:
        json.dump(token_cts, fd)
