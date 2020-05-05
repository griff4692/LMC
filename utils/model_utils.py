import os
import subprocess
import sys

import numpy as np


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def get_git_revision_hash():
    """
    :return: current git hash
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')


def render_args(args):
    """
    :param args: argparse instance
    :return: None
    Renders out argparse state key-value pairs to stdout
    """
    for arg in vars(args):
        print('{}={}'.format(arg, getattr(args, arg)))


def tensor_to_np(tens):
    """
    :param tens: tensor
    :return: numpy version of tensor
    Assumes tensor is on cpu but handles exception in case it's on cuda and needs to be converted to cpu first.
    """
    tens = tens.detach()
    try:
        return tens.numpy()
    except TypeError:
        return tens.cpu().numpy()


def render_num_params(model, vocab_size):
    pms = list(model.parameters())
    pm_size = 0
    full_pm_size = 0
    for pm in pms:
        size = list(pm.size())
        if size[0] < vocab_size:
            pm_size += np.array(size).prod()
        full_pm_size += np.array(size).prod()
    print('Model has {} parameters.  {} without counting embeddings'.format(full_pm_size, pm_size))
