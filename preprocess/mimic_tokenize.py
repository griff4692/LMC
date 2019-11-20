from collections import defaultdict
import json
import os

import re

from nltk import word_tokenize

import argparse
import pandas as pd


"""
Preprocess PubMed abstracts or MIMIC-III reports
"""


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


def preprocess_mimic_to_sentences(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()


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


def preprocess_pubmed(text):
    """
    Preprocess PubMed abstract. (https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PubMed/)
    1. tokenize sentences and words
    2. lowercase
    """
    for sent in sent_tokenize(text):
        text = ' '.join(word_tokenize(sent))
        yield text.lower()


def test_preprocess_mimic():
    text = """Normal sinus rhythm. Right bundle-branch block with secondary ST-T wave
    abnormalities. Compared to the previous tracing of [**2198-5-26**] no diagnostic
    interim change.

    """
    sents = [sen for sen in preprocess_mimic(text)]
    assert sents[0] == 'normal sinus rhythm .'
    assert sents[1] == 'right bundle-branch block with secondary st-t wave abnormalities .'
    assert sents[2] == 'compared to the previous tracing of no diagnostic interim change .'


def test_preprocess_pubmed():
    # from https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_xml/30096728/ascii

    text = 'A novel chiral porous-layer stationary phase was developed for use in open-tubular ' \
           'nano liquid chromatography. The stationary phase was prepared by an in-situ ' \
           'polymerization of 3-chloro-2-hydroxypropylmethacrylate (HPMA-Cl) and ethylene ' \
           'dimethacrylate (EDMA). The reactive chloro groups at the surface of the porous ' \
           'stationary phase were reacted with beta-Cyclodextrin (beta-CD). '
    sents = [sen for sen in preprocess_pubmed(text)]
    assert sents[2] == 'the reactive chloro groups at the surface of the porous stationary phase ' \
                       'were reacted with beta-cyclodextrin ( beta-cd ) .'




if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--mimic_fp', default='~/Desktop/mimic/NOTEEVENTS')

    args = arguments.parse_args()

    # Expand home path (~) so that pandas knows where to look
    print('Loading data...')
    args.mimic_fp = os.path.expanduser(args.mimic_fp)
    df = pd.read_csv(args.mimic_fp + '.csv')
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
        if (row_idx + 1) % 1000 == 0:
            print('Processed {} out of {} rows'.format(row_idx + 1, df.shape[0]))
    json.dump(list(zip(categories, parsed_docs)), open(args.mimic_fp + '_tokenized.json', 'w'))
    json.dump(token_cts, open(args.mimic_fp + '_token_counts.json', 'w'))
