import os

import argparse
import torch

from bsg_model import BSG


def restore_model(restore_name):
    checkpoint_dir = os.path.join('../bsg/weights', restore_name)
    checkpoint_fns = os.listdir(checkpoint_dir)
    checkpoint_fns = list(filter(lambda x: 'results' not in x, checkpoint_fns))
    max_checkpoint_epoch, latest_checkpoint_idx = -1, -1
    for cidx, checkpoint_fn in enumerate(checkpoint_fns):
        if 'best' in checkpoint_fn:
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
    vocab = checkpoint_state['vocab']
    print('Previous checkpoint at epoch={}...'.format(max_checkpoint_epoch))
    for k, v in checkpoint_state['losses'].items():
        print('{}={}'.format(k, v))
    args = argparse.ArgumentParser()
    for k, v in checkpoint_state['args'].items():
        print('{}={}'.format(k, v))
        setattr(args, k, v)
    vae_model = BSG(args, vocab.size())
    vae_model.load_state_dict(checkpoint_state['model_state_dict'])
    optimizer_state = checkpoint_state['optimizer_state_dict']
    return args, vae_model, vocab, optimizer_state


def save_checkpoint(args, model, optimizer, vocab, losses_dict, checkpoint_fp=None):
    # Serializing everything from model weights and optimizer state, to to loss function and arguments
    state_dict = {'model_state_dict': model.state_dict()}
    state_dict.update(losses_dict)
    state_dict.update({'optimizer_state_dict': optimizer.state_dict()})
    args_dict = {'args': {arg: getattr(args, arg) for arg in vars(args)}}
    state_dict.update(args_dict)
    state_dict.update({'vocab': vocab})
    # Serialize model and statistics
    print('Saving model state to {}'.format(checkpoint_fp))
    torch.save(state_dict, checkpoint_fp)
