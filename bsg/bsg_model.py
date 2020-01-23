import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from bsg_encoder import BSGEncoderLSTM
from compute_utils import compute_kl, mask_2D


class BSG(nn.Module):
    def __init__(self, args, vocab_size, input_dim=100):
        super(BSG, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = BSGEncoderLSTM(vocab_size)
        self.margin = args.hinge_loss_margin or 1.0

        # The output representations of words(used in KL regularization and max_margin).
        self.embeddings_mu = nn.Embedding(vocab_size, input_dim, padding_idx=0)

        self.embeddings_log_sigma = nn.Embedding(vocab_size, 1, padding_idx=0)
        log_weights_init = np.random.uniform(low=-3.5, high=-1.5, size=(vocab_size, 1))
        self.embeddings_log_sigma.load_state_dict({'weight': torch.from_numpy(log_weights_init)})

        self.multi_bsg = args.multi_bsg
        if hasattr(args, 'mask_p'):
            self.mask_p = args.mask_p
        else:
            self.mask_p = None
        if args.multi_bsg:
            if hasattr(args, 'multi_weights'):
                weights = np.array(list(map(float, args.multi_weights.split(','))))
            else:
                weights = np.array([0.7, 0.2, 0.1])
            self.input_weights = torch.from_numpy(weights).to(self.device)

    def _max_margin(self, mu_q, sigma_q, pos_mu_p, pos_sigma_p, neg_mu_p, neg_sigma_p, mask):
        """
        Computes a sum over context words margin(hinge loss).
        :param pos_context_words:  a tensor with true context words ids [batch_size x window_size]
        :param neg_context_words: a tensor with negative context words ids [batch_size x window_size]
        :param num_contexts: batch_size (how many context words for each center id - necessary for masking padding)
        :return: tensor [batch_size x 1]
        """
        batch_size, num_context_ids, embed_dim = pos_mu_p.size()

        mu_q_tiled = mu_q.unsqueeze(1).repeat(1, num_context_ids, 1)
        sigma_q_tiled = sigma_q.unsqueeze(1).repeat(1, num_context_ids, 1)
        mu_q_flat = mu_q_tiled.view(batch_size * num_context_ids, -1)
        sigma_q_flat = sigma_q_tiled.view(batch_size * num_context_ids, -1)

        pos_mu_p_flat = pos_mu_p.view(batch_size * num_context_ids, -1)
        pos_sigma_p_flat = pos_sigma_p.view(batch_size * num_context_ids, -1)
        neg_mu_p_flat = neg_mu_p.view(batch_size * num_context_ids, -1)
        neg_sigma_p_flat = neg_sigma_p.view(batch_size * num_context_ids, -1)

        kl_pos = compute_kl(mu_q_flat, sigma_q_flat, pos_mu_p_flat, pos_sigma_p_flat, device=self.device).view(
            batch_size, -1)
        kl_neg = compute_kl(mu_q_flat, sigma_q_flat, neg_mu_p_flat, neg_sigma_p_flat, device=self.device).view(
            batch_size, -1)

        hinge_loss = (kl_pos - kl_neg + self.margin).clamp_min_(0)
        hinge_loss.masked_fill_(mask, 0)
        return hinge_loss.sum(1)

    def _compute_priors(self, ids):
        return self.embeddings_mu(ids), self.embeddings_log_sigma(ids).exp()

    def forward(self, token_ids, sec_ids, cat_ids, context_ids, neg_context_ids, num_contexts):
        """
        :param center_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :param neg_context_ids: batch_size, 2 * context_window
        :param num_contexts: batch_size (how many context words for each center id - necessary for masking padding)
        :return: cost components: KL-Divergence (q(z|w,c) || p(z|w)) and max margin (reconstruction error)
        """
        # Mask padded context ids
        batch_size, num_context_ids = context_ids.size()
        mask_size = torch.Size([batch_size, num_context_ids])
        mask = mask_2D(mask_size, num_contexts).to(self.device)

        center_ids = token_ids
        if self.multi_bsg:
            center_id_candidates = torch.cat([
                token_ids.unsqueeze(0),
                sec_ids.unsqueeze(0),
                cat_ids.unsqueeze(0)
            ])
            input_sample = torch.multinomial(self.input_weights, batch_size, replacement=True).to(self.device)
            center_ids = center_id_candidates.gather(0, input_sample.unsqueeze(0)).squeeze(0)

        mu_q, sigma_q = self.encoder(center_ids, context_ids, mask, token_mask_p=self.mask_p)
        mu_p, sigma_p = self._compute_priors(token_ids)

        pos_mu_p, pos_sigma_p = self._compute_priors(context_ids)
        neg_mu_p, neg_sigma_p = self._compute_priors(neg_context_ids)

        kl = compute_kl(mu_q, sigma_q, mu_p, sigma_p).mean()
        max_margin = self._max_margin(mu_q, sigma_q, pos_mu_p, pos_sigma_p, neg_mu_p, neg_sigma_p, mask).mean()
        return kl, max_margin
