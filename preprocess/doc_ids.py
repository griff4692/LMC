import numpy as np


def parse_doc_ids(vocab, ids, doc2vec=True):
    # TODO make this part of preprocessing
    token_vocab_size = vocab.size()

    # The document boundary indices (pad_idx = 0) is never going to be a center word
    doc_pos_idxs = np.where(ids == 0)[0]

    # We don't need to increase vocabulary size with document ids since they won't be modeled
    # We only need to know where they are so we don't use them as center words
    if not doc2vec:
        return doc_pos_idxs, [], []

    doc_ids = []
    for doc_num, doc_pos_idx in enumerate(doc_pos_idxs):
        doc_id = vocab.add_token('doc@ids[{}]'.format(doc_pos_idx))
        if doc_num + 1 == len(doc_pos_idxs):
            doc_len = len(ids) - doc_pos_idx
        else:
            doc_len = doc_pos_idxs[doc_num + 1] - doc_pos_idx
        doc_ids += [doc_id] * doc_len
        ids[doc_pos_idx] = doc_id  # Make doc_ids be negative so we know difference between word and document easily
    full_vocab_size = vocab.size()
    # Documents are stored at the end of the vocabulary
    doc_id_range = np.arange(token_vocab_size, full_vocab_size)

    return doc_pos_idxs, doc_ids, doc_id_range
