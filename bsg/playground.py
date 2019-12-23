import argparse
import torch

from compute_utils import compute_kl
from model_utils import restore_model, tensor_to_np
from vae import VAE

EXTRA_CONTEXTS = 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('--experiment', default='baseline-v2', help='Save path in weights/ for experiment.')

    args = parser.parse_args()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    prev_args, vae_model, vocab, optimizer_state = restore_model(args.experiment)

    # Make sure it's NOT calculating gradients
    model = vae_model.to(args.device)
    model.eval()  # just sets .requires_grad = False

    # Setup problem
    window_toks = 'bleeding cervical os consistent incomplete patient presents suction & c'
    window_toks = window_toks.split()
    full_tokens = 'She is now bleeding quite heavily. Ultrasound this morning demonstrated a missed AB consistent with a 6 week pregnancy with blood clots in the uterine cavity, as well as continued bleeding from the cervical os. This is consistent with an incomplete AB. The patient presents now for a suction D&C. Medical history is negative. Surgical history is negative. CURRENT MEDICATIONS: Include prenatal vitamins.",bleeding quite heavily ultrasound morning demonstrated missed ab consistent week pregnancy blood clots uterine cavity well continued bleeding cervical os consistent incomplete ab patient presents suction & c medical history negative surgical history negative current medications include prenatal vitamins'
    full_tokens = list(set(full_tokens.split()))
    pos_center = ['abortion']
    neg_center = ['blood']
    sf = 'ab'
    # End of setup

    full_priors = {'mu': [], 'sigma': [], 'token': []}
    for token in full_tokens:
        center_id = [vocab.get_id(token.lower())]
        mu, sigma = model._compute_priors(torch.LongTensor(center_id).clamp_min(0))
        full_priors['mu'].append(mu)
        full_priors['sigma'].append(sigma)

    full_priors['mu'] = torch.cat(full_priors['mu'], axis=0)
    full_priors['sigma'] = torch.cat(full_priors['sigma'], axis=0)

    ab_tens = torch.LongTensor([vocab.get_id(sf)]).clamp_min_(0)

    neg_center_id = [vocab.get_id(neg_center[0])]
    pos_center_id = [vocab.get_id(pos_center[0])]

    pos_center_tens = torch.LongTensor(pos_center_id).clamp_min(0)
    neg_center_tens = torch.LongTensor(neg_center_id).clamp_min(0)
    pos_mu, pos_sigma = model._compute_priors(pos_center_tens)
    neg_mu, neg_sigma = model._compute_priors(neg_center_tens)

    window_tens = torch.LongTensor([vocab.get_id(t) for t in window_toks]).clamp_min(0).unsqueeze(0)
    window_mask = torch.BoolTensor(torch.Size([1, len(window_toks)]))
    window_mask.fill_(0)

    with torch.no_grad():
        z_orig_mu, z_orig_sigma = model.encoder(ab_tens, window_tens, window_mask)

    orig_kl_pos = compute_kl(z_orig_mu, z_orig_sigma, pos_mu, pos_sigma).item()
    orig_kl_neg = compute_kl(z_orig_mu, z_orig_sigma, neg_mu, neg_sigma).item()

    extra_context_tokens = []
    extra_kls = []
    for idx, (mu, sigma) in enumerate([(pos_mu, pos_sigma), (neg_mu, neg_sigma)]):
        lf = pos_center[0] if idx == 0 else neg_center[0]

        posterior_kld_full_tokens = compute_kl(mu, sigma, full_priors['mu'], full_priors['sigma']).squeeze(1)
        closest_token_idxs = posterior_kld_full_tokens.squeeze().argsort()[:EXTRA_CONTEXTS].numpy()
        closest_tokens = []
        for idx in closest_token_idxs:
            closest_tokens.append(full_tokens[idx])
        extra_context_tokens.append(closest_tokens)

        print('For LF={}, added tokens to context={}'.format(lf, ', '.join(closest_tokens)))

        desired_window_toks = list(set(closest_tokens + window_toks))
        desired_tens = torch.LongTensor([vocab.get_id(t) for t in desired_window_toks]).clamp_min(0).unsqueeze(0)
        desired_mask = torch.BoolTensor(torch.Size([1, len(desired_window_toks)]))
        desired_mask.fill_(0)

        with torch.no_grad():
            z_desired_mu, z_desired_sigma = model.encoder(ab_tens, desired_tens, desired_mask)

        desired_kl = compute_kl(z_desired_mu, z_desired_sigma, mu, sigma).item()
        extra_kls.append(desired_kl)

    print('Given context window divergences --> Target={}, Sampled LF={}.'.format(orig_kl_pos, orig_kl_neg))
    print('Desired context window divergences --> Target={}, Sampled LF={}.'.format(extra_kls[0], extra_kls[1]))
