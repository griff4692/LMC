import json
import pickle

import numpy as np
from tqdm import tqdm


def tokens_to_ids(args, token_infile):
    """"
    :param args: argparse instance
    :param token_infile: Where to read the tokens from
    :return: None

    Dumps ids.npy and vocab.pk to preprocess/data/
    These are the two data structures used for generating training examples.
    ids.npy -> a flattened list of subsampled token ids for the whole corpus
    vocab.pk -> vocabulary for tokens and metadata (section and note category)
    """
    debug_str = '_mini' if args.debug else ''

    with open(token_infile, 'r') as fd:
        tokens = json.load(fd)

    # Load Vocabulary
    vocab_infile = 'data/mimic/vocab{}.pk'.format(debug_str)
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    ids = []
    N = len(tokens)
    for doc_idx in tqdm(range(N)):
        ids += vocab.get_ids(tokens[doc_idx].split())

    print('Saving {} tokens to disc'.format(len(ids)))
    out_fn = 'data/mimic/ids{}.npy'.format(debug_str)
    with open(out_fn, 'wb') as fd:
        np.save(fd, np.array(ids, dtype=int))
    with open(vocab_infile, 'wb') as fd:
        pickle.dump(vocab, fd)
