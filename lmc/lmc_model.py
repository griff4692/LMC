import sys

import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from lmc_encoder import LMCEncoder
from compute_utils import compute_kl, mask_2D


class LMC(nn.Module):
    def __init__(self, args, token_vocab_size, section_vocab_size):
        super(LMC, self).__init__()
        self.device = args.device
        self.encoder = LMCEncoder(args, token_vocab_size, section_vocab_size)
        self.margin = args.hinge_loss_margin or 1.0

    def _compute_marginal(self, ids, section_ids, section_p):
        ids_tiled = ids.unsqueeze(-1).repeat(1, 1, section_ids.size()[-1])
        mu_marginal, sigma_marginal = self.encoder(ids_tiled, section_ids)
        section_p = section_p.unsqueeze(-1)
        mu = (section_p * mu_marginal).sum(2)
        sigma = (section_p * sigma_marginal).sum(2)
        return mu, sigma + 1e-3

    def forward(self, center_ids, center_section_ids, context_ids, context_section_ids, neg_ids, neg_section_ids,
                num_contexts, context_section_p=None, neg_section_p=None):
        """
        :param center_ids: batch_size
        :param center_section_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :param context_section_ids: batch_size, 2 * context_window, max_num_sections
        :param neg_ids: batch_size, 2 * context_window
        :param neg_section_ids: batch_size, 2 * context_window, max_num_sections
        :param num_contexts: batch_size (how many context words for each center id - necessary for masking padding)
        :param context_section_p: batch_size, 2 * context_window, max_num_sections
        :param neg_section_p: batch_size, 2 * context_window, max_num_sections
        :return: cost components: KL-Divergence (q(z|w,c) || p(z|w)) and max margin (reconstruction error)
        """
        # Mask padded context ids
        batch_size, num_context_ids = context_ids.size()
        mask_size = torch.Size([batch_size, num_context_ids])
        mask = mask_2D(mask_size, num_contexts).to(self.device)

        # Compute center words
        mu_center, sigma_center = self.encoder(center_ids, center_section_ids)
        mu_center_tiled = mu_center.unsqueeze(1).repeat(1, num_context_ids, 1)
        sigma_center_tiled = sigma_center.unsqueeze(1).repeat(1, num_context_ids, 1)
        mu_center_flat = mu_center_tiled.view(batch_size * num_context_ids, -1)
        sigma_center_flat = sigma_center_tiled.view(batch_size * num_context_ids, -1)

        # Compute positive and negative encoded samples
        if context_section_p is None and neg_section_p is None:
            mu_pos_context, sigma_pos_context = self.encoder(context_ids, context_section_ids)
            mu_neg_context, sigma_neg_context = self.encoder(neg_ids, neg_section_ids)
        else:
            mu_pos_context, sigma_pos_context = self._compute_marginal(
                context_ids, context_section_ids, context_section_p)
            mu_neg_context, sigma_neg_context = self._compute_marginal(
                neg_ids, neg_section_ids, neg_section_p)

        # Flatten positive context
        mu_pos_context_flat = mu_pos_context.view(batch_size * num_context_ids, -1)
        sigma_pos_context_flat = sigma_pos_context.view(batch_size * num_context_ids, -1)

        # Flatten negative context
        mu_neg_context_flat = mu_neg_context.view(batch_size * num_context_ids, -1)
        sigma_neg_context_flat = sigma_neg_context.view(batch_size * num_context_ids, -1)

        # Compute KL-divergence between center words and negative and reshape
        kl_pos_flat = compute_kl(mu_center_flat, sigma_center_flat, mu_pos_context_flat, sigma_pos_context_flat)
        kl_neg_flat = compute_kl(mu_center_flat, sigma_center_flat, mu_neg_context_flat, sigma_neg_context_flat)
        kl_pos = kl_pos_flat.view(batch_size, num_context_ids)
        kl_neg = kl_neg_flat.view(batch_size, num_context_ids)

        hinge_loss = (kl_pos - kl_neg + self.margin).clamp_min_(0)
        hinge_loss.masked_fill_(mask, 0)
        hinge_loss = hinge_loss.sum(1)

        return hinge_loss.mean()
