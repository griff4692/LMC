import pickle

from transformers import BertTokenizer

SPECIAL_TOKS = ['[UNK]', '[PAD]', '[MASK]', '[CLS]', '[SEP]']


def _standardize(str):
    str = str.strip()
    if str not in SPECIAL_TOKS:
        str = str.lower()
    return str


if __name__ == '__main__':
    with open('data/vocab.pk', 'rb') as fd:
        vocab = pickle.load(fd)

    split_pt = min(vocab.section_start_vocab_id, vocab.category_start_vocab_id)
    metadata_tokens = vocab.i2w[split_pt:] + ['digitparsed']

    with open('data/clinic_bert_vocab.txt', 'r') as fd:
        clinic_bert_tokens = fd.readlines()

    clinic_bert_tokens = list(set(list(map(_standardize, clinic_bert_tokens))))
    vocab_fn = 'data/clinic_bert_plus_metadata_vocab.txt'
    with open(vocab_fn, 'w') as fd:
        fd.write('\n'.join(clinic_bert_tokens + metadata_tokens))
    out_fn = 'data/tokenizer_vocab.pth'
    print('Creating BERT tokenizer and saving to {}'.format(out_fn))
    tokenizer = BertTokenizer(vocab_fn, never_split=metadata_tokens, do_basic_tokenize=False)

    special_tokens_dict = {'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]', 'bos_token': '[BOS]',
                           'eos_token': '[EOS]', 'pad_token': '[PAD]', 'mask_token': '[MASK]',
                           'additional_special_tokens': metadata_tokens}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')

    example_sentence = 'header=HPI example clinical text with tricky autoimmunihistory word'
    print('Tokenization of sentence: {}'.format(example_sentence))
    tokenizer.save_vocabulary(out_fn)
