from collections import OrderedDict
import os
import sys

import argparse
import torch
from transformers import BertTokenizer

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'modules', 'lmc'))
from lmc_model import LMC, LMCBERT


def load_bert_tokenizer(bert_tokenizer, metadata_vocab):
    metadata_tokens = metadata_vocab.i2w[1:] + ['digitparsed']
    special_tokens_dict = {'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]', 'bos_token': '[BOS]',
                           'eos_token': '[EOS]', 'pad_token': '[PAD]', 'mask_token': '[MASK]',
                           'additional_special_tokens': metadata_tokens}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)
    print('Readded special tokens={}'.format(num_added_toks))
    print('Mapping regular vocab ids to WordPiece ids for token and metadata...')


def restore_model(restore_name):
    """
    :param restore_name: Directory name within ~/LMC/weights/lmc where model weights are serialized
    :return: state of the previously trained model, including model weights, training arguments, and vocabularies used
    """
    checkpoint_dir = os.path.join(home_dir, 'weights', 'lmc', restore_name)
    checkpoint_fns = os.listdir(checkpoint_dir)
    checkpoint_fns = list(filter(lambda x: 'results' not in x, checkpoint_fns))
    max_checkpoint_epoch, latest_checkpoint_idx = -1, -1
    for cidx, checkpoint_fn in enumerate(checkpoint_fns):
        if 'best' in checkpoint_fn:  # Always select 'best' if it exists in weights directory
            latest_checkpoint_idx = cidx
            break
        checkpoint_epoch = int(checkpoint_fn.split('_')[-1].split('.')[0])
        max_checkpoint_epoch = max(max_checkpoint_epoch, checkpoint_epoch)
        if checkpoint_epoch == max_checkpoint_epoch:
            latest_checkpoint_idx = cidx
    latest_checkpoint_fn = os.path.join(checkpoint_dir, checkpoint_fns[latest_checkpoint_idx])
    print('Loading model from {}'.format(latest_checkpoint_fn))
    if not torch.cuda.is_available():
        checkpoint_state = torch.load(latest_checkpoint_fn, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(latest_checkpoint_fn)
    token_vocab, metadata_vocab = checkpoint_state['token_vocab'], checkpoint_state['metadata_vocab']
    print('Previous checkpoint at epoch={}...'.format(max_checkpoint_epoch))
    for k, v in checkpoint_state['losses'].items():
        print('{}={}'.format(k, v))
    args = argparse.ArgumentParser()
    for k, v in checkpoint_state['args'].items():
        print('{}={}'.format(k, v))
        setattr(args, k, v)
    new_state_dict = OrderedDict()
    # When using nn.DataParallel(model) it prepends 'module.' to all parameter names in state dict.
    # Need to remove before loading state dict.
    for k, v in checkpoint_state['model_state_dict'].items():
        name = k
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    token_vocab_size = token_vocab.size()
    bert_tokenizer = None
    if hasattr(args, 'bert') and args.bert:
        bert_tokenizer = checkpoint_state['bert_tokenizer']

        # TODO remove.  This is for backward compatibility
        if bert_tokenizer is None:
            bert_tokenizer_fn = os.path.join(home_dir, 'preprocess/data/bert_tokenizer_vocab.pth')
            bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_fn)
            load_bert_tokenizer(bert_tokenizer, metadata_vocab)
        token_vocab_size = max(bert_tokenizer.vocab_size, max(bert_tokenizer.all_special_ids) + 1)
        lmc_model = LMCBERT(args, token_vocab_size)
    else:
        lmc_model = LMC(args, token_vocab_size, metadata_vocab.size())
    lmc_model.load_state_dict(new_state_dict)

    optimizer_state = checkpoint_state['optimizer_state_dict']
    token_metadata_counts = checkpoint_state['token_metadata_counts']
    return args, lmc_model, token_vocab, metadata_vocab, bert_tokenizer, optimizer_state, token_metadata_counts


def save_checkpoint(args, model, optimizer, token_vocab, losses_dict, token_metadata_counts=None,
                    checkpoint_fp=None, metadata_vocab=None, bert_tokenizer=None):
    # Serializes everything from model weights and optimizer state, to loss function and arguments
    state_dict = {'model_state_dict': model.state_dict(), 'token_metadata_counts': token_metadata_counts}
    state_dict.update(losses_dict)
    state_dict.update({'optimizer_state_dict': optimizer.state_dict()})
    args_dict = {'args': {arg: getattr(args, arg) for arg in vars(args)}}
    state_dict.update(args_dict)
    state_dict.update({'token_vocab': token_vocab})
    state_dict.update({'metadata_vocab': metadata_vocab})
    state_dict.update({'bert_tokenizer': bert_tokenizer})
    # Serialize model and statistics
    print('Saving model state to {}'.format(checkpoint_fp))
    torch.save(state_dict, checkpoint_fp)
