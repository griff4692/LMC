import os

import argparse
import torch

from vae import VAE


def tensor_to_np(tens):
    tens = tens.detach()
    try:
        return tens.numpy()
    except TypeError:
        return tens.cpu().numpy()


def restore_model(vocab_size, restore_name):
    checkpoint_dir = os.path.join('weights', restore_name)
    checkpoint_fns = os.listdir(checkpoint_dir)
    max_checkpoint_epoch, latest_checkpoint_idx = -1, -1
    for cidx, checkpoint_fn in enumerate(checkpoint_fns):
        checkpoint_epoch = int(checkpoint_fn[-5])
        max_checkpoint_epoch = max(max_checkpoint_epoch, checkpoint_epoch)
        if checkpoint_epoch == max_checkpoint_epoch:
            latest_checkpoint_idx = cidx
    latest_checkpoint_fn = os.path.join(checkpoint_dir, checkpoint_fns[latest_checkpoint_idx])
    print('Loading model from {}'.format(latest_checkpoint_fn))
    checkpoint_state = torch.load(latest_checkpoint_fn)
    print('Previous checkpoint at epoch={}...'.format(max_checkpoint_epoch))
    for k, v in checkpoint_state['losses'].items():
        print('{}={}'.format(k, v))
    args = argparse.ArgumentParser()
    for k, v in checkpoint_state['args'].items():
        print('{}={}'.format(k, v))
        setattr(args, k, v)

    vae_model = VAE(args, vocab_size)
    vae_model.load_state_dict(checkpoint_state['model_state_dict'])

    return args, vae_model, checkpoint_state['optimizer_state_dict']
