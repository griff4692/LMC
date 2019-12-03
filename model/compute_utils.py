import torch


def compute_kl(mu_a, sigma_a, mu_b, sigma_b):
    """
    :param mu_a: mean vector of batch_size x dim
    :param sigma_a: standard deviation of batch_size x 1
    :param mu_b: mean vector of batch_size x dim
    :param sigma_b: standard deviation of batch_size x 1
    :return: computes KL-Divergence between 2 spherical Gaussian (a||b)
    """
    d = mu_a.shape[1]
    sigma_p_inv = 1.0 / sigma_b  # because diagonal
    tra = d * sigma_a * sigma_p_inv
    quadr = sigma_p_inv * torch.pow(mu_b - mu_a, 2).sum(1, keepdim=True)
    log_det = - d * torch.log(sigma_a * sigma_p_inv)
    res = 0.5 * (tra + quadr - d + log_det)
    return res
