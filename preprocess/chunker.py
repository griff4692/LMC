import os
import re

import numpy as np
from quickumls import QuickUMLS

QUICKUMLS_FP = os.path.expanduser('~/quickumls_data/')
matcher = QuickUMLS(QUICKUMLS_FP, threshold=0.7, similarity_name='jaccard', window=5)


def parse_chunker(original_text, phrase_matches):
    order = np.argsort([match[0]['start'] for match in phrase_matches])

    offset = 0
    chunked_string = original_text
    prev_end = 0
    for num_match, match_idx in enumerate(order):
        match = phrase_matches[match_idx]
        ngram = match[0]['ngram']
        term = match[0]['term']
        start = match[0]['start']
        end = match[0]['end']
        assert start >= prev_end
        prev_end = end
        # Only change multi word phrases
        if len(ngram.split()) == 1:
            continue
        term = '_'.join(term.split())
        if term == 'x_medicine':
            continue
        chunked_string = chunked_string[:start + offset] + ' ' + term + ' ' + chunked_string[end + offset:]
        offset += len(term) - len(ngram) + 2
    return re.sub(r'\s+', ' ', chunked_string)


def chunk(doc):
    phrases = matcher.match(doc, best_match=True, ignore_syntax=False)
    return parse_chunker(doc, phrases)


if __name__ == '__main__':
    text = 'breast cancer'
    print(chunk([text]))
