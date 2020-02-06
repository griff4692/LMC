import sys

import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from lmc_context_encoder import LMCContextEncoder
from lmc_context_decoder import LMCDecoder
from compute_utils import compute_kl, mask_2D


class LMCC(nn.Module):
    def __init__(self, args, token_vocab_size, section_vocab_size):
        super(LMCC, self).__init__()
        self.device = args.device
        self.encoder = LMCContextEncoder(token_vocab_size, section_vocab_size)
        self.decoder = LMCDecoder(token_vocab_size, section_vocab_size)

        self.margin = args.hinge_loss_margin or 1.0

    def _compute_marginal(self, ids, metadata_ids):
        ids_tiled = ids.unsqueeze(-1).repeat(1, 1, metadata_ids.size()[-1])
        mu_marginal, sigma_marginal = self.decoder(ids_tiled, metadata_ids)
        # mu = mu_marginal.mean(2)
        # sigma = sigma_marginal.mean(2)
        return mu_marginal, sigma_marginal + 1e-3

    def forward(self, center_ids, center_metadata_ids, context_ids, context_metadata_ids, neg_ids, neg_metadata_ids,
                num_contexts):
        """
        :param center_ids: batch_size
        :param center_metadata_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :param context_metadata_ids: batch_size, 2 * context_window, max_num_sections
        :param neg_ids: batch_size, 2 * context_window
        :param neg_metadata_ids: batch_size, 2 * context_window, max_num_sections
        :param num_contexts: batch_size (how many context words for each center id - necessary for masking padding)
        :return: cost components: KL-Divergence (q(z|w,c) || p(z|w)) and max margin (reconstruction error)
        """
        # Mask padded context ids
        batch_size, num_context_ids = context_ids.size()
        mask_size = torch.Size([batch_size, num_context_ids])
        mask = mask_2D(mask_size, num_contexts).to(self.device)

        m_samples = 10

        # Compute center words
        mu_center_q, sigma_center_q, _ = self.encoder(center_ids, center_metadata_ids, context_ids, mask)
        mu_center_tiled_q = mu_center_q.unsqueeze(1).repeat(1, num_context_ids * m_samples, 1)
        sigma_center_tiled_q = sigma_center_q.unsqueeze(1).repeat(1, num_context_ids * m_samples, 1)
        mu_center_flat_q = mu_center_tiled_q.view(batch_size * num_context_ids * m_samples, -1)
        sigma_center_flat_q = sigma_center_tiled_q.view(batch_size * num_context_ids * m_samples, -1)

        # Compute decoded representations of (w, d), E(c), E(n)
        mu_center, sigma_center = self.decoder(center_ids, center_metadata_ids)
        mu_pos_context, sigma_pos_context = self._compute_marginal(context_ids, context_metadata_ids)
        mu_neg_context, sigma_neg_context = self._compute_marginal(neg_ids, neg_metadata_ids)

        # Flatten positive context
        mu_pos_context_flat = mu_pos_context.view(batch_size * num_context_ids * m_samples, -1)
        sigma_pos_context_flat = sigma_pos_context.view(batch_size * num_context_ids * m_samples, -1)

        # Flatten negative context
        mu_neg_context_flat = mu_neg_context.view(batch_size * num_context_ids * m_samples, -1)
        sigma_neg_context_flat = sigma_neg_context.view(batch_size * num_context_ids * m_samples, -1)

        # Compute KL-divergence between center words and negative and reshape
        kl_pos_flat = compute_kl(mu_center_flat_q, sigma_center_flat_q, mu_pos_context_flat, sigma_pos_context_flat)
        kl_neg_flat = compute_kl(mu_center_flat_q, sigma_center_flat_q, mu_neg_context_flat, sigma_neg_context_flat)
        kl_pos = kl_pos_flat.view(batch_size, num_context_ids, m_samples).mean(-1)
        kl_neg = kl_neg_flat.view(batch_size, num_context_ids, m_samples).mean(-1)

        hinge_loss = (kl_pos - kl_neg + self.margin).clamp_min_(0)
        hinge_loss.masked_fill_(mask, 0)
        hinge_loss = hinge_loss.sum(1)

        recon_loss = compute_kl(mu_center_q, sigma_center_q, mu_center, sigma_center).squeeze(-1)
        return hinge_loss.mean(), recon_loss.mean()
