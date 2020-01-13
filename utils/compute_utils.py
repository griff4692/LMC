import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.nn as nn


def compute_att(h, mask, att_linear):
    att_scores = att_linear(h).squeeze(-1)
    att_scores.masked_fill_(mask, -1e5)
    att_weights = nn.Softmax(1)(att_scores)
    h_sum = (att_weights.unsqueeze(-1) * h).sum(1)
    return h_sum


def compute_kl(mu_a, sigma_a, mu_b, sigma_b, device=None):
    """
    :param mu_a: mean vector of batch_size x dim
    :param sigma_a: standard deviation of batch_size x {1, dim}
    :param mu_b: mean vector of batch_size x dim
    :param sigma_b: standard deviation of batch_size x {1, dim}
    :return: computes KL-Divergence between 2 diagonal Gaussian (a||b)
    """
    var_dim = sigma_a.size()[-1]
    assert sigma_b.size()[-1] == var_dim
    if var_dim == 1:
        return kl_spher(mu_a, sigma_a, mu_b, sigma_b)
    return kl_diag(mu_a, sigma_a, mu_b, sigma_b, device=device)


def kl_spher(mu_a, sigma_a, mu_b, sigma_b):
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


def kl_diag(mu_a, sigma_a, mu_b, sigma_b, device=None):
    """
    :param mu_a: mean vector of batch_size x dim
    :param sigma_a: standard deviation of batch_size x dim
    :param mu_b: mean vector of batch_size x dim
    :param sigma_b: standard deviation of batch_size x dim
    :return: computes KL-Divergence between 2 diagonal Gaussians (a||b)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, components = mu_a.size()
    _, var_dim = sigma_a.size()

    # Create diagonal covariance matrix
    if var_dim == 1:
        sigma_a_tiled = sigma_a.repeat(1, components)
        sigma_b_tiled = sigma_b.repeat(1, components)
    else:
        sigma_a_tiled = sigma_a
        sigma_b_tiled = sigma_b

    a_covar = torch.zeros(batch_size, components, components).to(device)
    b_covar = a_covar.clone()
    a_covar[:, range(components), range(components)] = sigma_a_tiled

    b_covar[:, range(components), range(components)] = sigma_b_tiled

    dist_a = MultivariateNormal(mu_a, covariance_matrix=a_covar)
    dist_b = MultivariateNormal(mu_b, covariance_matrix=b_covar)

    return kl_divergence(dist_a, dist_b)


def mask_2D(target_size, num_contexts):
    mask = torch.BoolTensor(target_size)
    mask.fill_(0)
    for batch_idx, num_c in enumerate(num_contexts):
        if num_c < target_size[1]:
            mask[batch_idx, num_c:] = 1
    return mask
