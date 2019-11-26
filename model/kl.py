import torch

def KL(mu_q, sigma_q, mu_p, sigma_p):
    
    
    d = mu_q.shape[1]
    sigma_p_inv = (1.0/sigma_p) #because diagonal
    tra = d * sigma_q*sigma_p_inv
    quadr = sigma_p_inv * torch.sum((mu_p - mu_q)**2, axis=1)
    log_det = - d*torch.log(sigma_q * sigma_p_inv)
    res = 0.5 * (tra + quadr - d + log_det)
    return res.reshape((-1, ))


