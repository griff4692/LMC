import json
import pickle
import re

import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-combine_phrases', default=False, action='store_true')

    args = parser.parse_args()
    # Load Data
    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''
    data_fp = '../../../preprocess/data/mimic/NOTEEVENTS_tokenized_subsampled{}{}.json'.format(debug_str, phrase_str)
    with open(data_fp, 'rb') as fd:
        data = json.load(fd)
    sentences = []

    for category, doc in data:
        doc_sentences = re.split(r'\bheader=SENTENCESTART|header=SENTENCEEND\b', doc)
        for sentence in doc_sentences:
            sentence = sentence.strip()
            tokens = sentence.split()
            if len(tokens) > 1 and len(list(filter(lambda x: 'header' not in x, tokens))) > 0:
                sentences.append(sentence)
    sentences = np.array(sentences)
    np.random.shuffle(sentences)
    with open('data/sentences{}{}.txt'.format(debug_str, phrase_str), 'w') as fd:
        fd.write('\n'.join(list(sentences)))
