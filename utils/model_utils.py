import subprocess


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
