from collections import OrderedDict
import os
import sys

import argparse
import torch

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'modules', 'lmc'))
from lmc_model import LMC


def restore_model(restore_name, ckpt=None):
    """
    :param restore_name: Directory name within ~/LMC/weights/lmc where model weights are serialized
    :param ckpt: Optional pre-specified epoch from which to restore checkpoint
    :return: state of the previously trained model, including model weights, training arguments, and vocabularies used
    """
    checkpoint_dir = os.path.join(home_dir, 'weights', 'lmc', restore_name)
    checkpoint_fns = os.listdir(checkpoint_dir)
    checkpoint_fns = list(filter(lambda x: 'results' not in x and 'metrics' not in x, checkpoint_fns))
    max_checkpoint_epoch, latest_checkpoint_idx = -1, -1
    for cidx, checkpoint_fn in enumerate(checkpoint_fns):
        checkpoint_epoch = int(checkpoint_fn.split('_')[-1].split('.')[0])
        if ckpt is not None and checkpoint_epoch == int(ckpt):
            latest_checkpoint_idx = cidx
            break
        if 'best' in checkpoint_fn and ckpt is None:  # Always select 'best' if it exists in weights directory
            latest_checkpoint_idx = cidx
            break
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
        token_vocab_size = max(bert_tokenizer.vocab_size, max(bert_tokenizer.all_special_ids) + 1)

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
