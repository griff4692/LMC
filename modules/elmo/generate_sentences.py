import json
import os
import re
import shutil

import argparse
from tqdm import tqdm

home_dir = os.path.expanduser('~/LMC/')


if __name__ == '__main__':
    """
    Necessary for training ELMo with AllenNLP module which expects a corpus file where each line is a unique sentence.
    
    Before running, you must run pre-processing pipeline (tokenize, subsample_tokens)
    with flag split_sentences=True.
    """
    parser = argparse.ArgumentParser('Generates list of sentences from pre-processed MIMIC data.')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)

    args = parser.parse_args()
    # Load Data
    debug_str = '_mini' if args.debug else ''
    print('Loading subsampled data...')
    data_fp = os.path.join(
        home_dir, 'preprocess/data/mimic/NOTEEVENTS_tokenized_subsampled{}_sentence.json'.format(debug_str))
    with open(data_fp, 'rb') as fd:
        data = json.load(fd)
    sentences = []
    chunk = 0
    print('Separating sentences...')
    for i in tqdm(range(len(data))):
        _, doc = data[i]
        doc_sentences = re.split(r'\bheader=SENTENCES|header=SENTENCESTART|header=SENTENCEEND\b', doc)
        for sentence in doc_sentences:
            sentence = sentence.strip()
            tokens = sentence.split()
            if len(tokens) > 1 and len(list(filter(lambda x: 'header' not in x, tokens))) > 0:
                sentences.append(sentence)
                if len(sentences) % 1000000 == 0:
                    with open('data/chunks/sentences{}_{}.txt'.format(debug_str, chunk), 'w') as fd:
                        fd.write('\n'.join(list(sentences)))
                    sentences = []
                    chunk += 1
    if len(sentences) > 0:
        with open('data/sentences{}_{}.txt'.format(debug_str, chunk), 'w') as fd:
            fd.write('\n'.join(list(sentences)))
        sentences = []
        chunk += 1
    with open('data/sentences{}.txt'.format(debug_str), 'a') as out_fd:
        for cidx in range(chunk):
            with open('data/chunks/sentences{}_{}.txt'.format(debug_str, cidx), 'r') as in_fd:
                shutil.copyfileobj(in_fd, out_fd)
