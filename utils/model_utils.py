import subprocess


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')


def render_args(args):
    for arg in vars(args):
        print('{}={}'.format(arg, getattr(args, arg)))


def tensor_to_np(tens):
    tens = tens.detach()
    try:
        return tens.numpy()
    except TypeError:
        return tens.cpu().numpy()
