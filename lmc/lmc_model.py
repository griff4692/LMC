import sys

import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_utils import compute_kl, mask_2D
from lmc_decoder import LMCDecoder, LMCDecoderBERT
from lmc_encoder import LMCEncoder, LMCEncoderBERT


class LMCBERT(nn.Module):
    def __init__(self, args, wp_vocab_size, metadata_vocab_size=None):
        super(LMCBERT, self).__init__()
        self.encoder = LMCEncoderBERT(args, wp_vocab_size)
        self.decoder = LMCDecoderBERT(args, wp_vocab_size)

    def forward(self, context_ids, center_ids, pos_ids, neg_ids, context_token_type_ids,
                num_contexts, context_mask, center_mask, pos_mask, neg_mask,
                tokenizer=None, num_metadata_samples=None):
        batch_size, num_context_ids, max_decoder_len = pos_ids.size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask_size = torch.Size([batch_size, num_context_ids])
        mask = mask_2D(mask_size, num_contexts).to(device)

        # Compute center words
        mu_center_q, sigma_center_q, _ = self.encoder(
            input_ids=context_ids, attention_mask=context_mask, token_type_ids=context_token_type_ids)
        mu_center_tiled_q = mu_center_q.unsqueeze(1).repeat(1, num_context_ids, 1)
        sigma_center_tiled_q = sigma_center_q.unsqueeze(1).repeat(1, num_context_ids, 1)
        mu_center_flat_q = mu_center_tiled_q.view(batch_size * num_context_ids, -1)
        sigma_center_flat_q = sigma_center_tiled_q.view(batch_size * num_context_ids, -1)

        # Compute decoded representations of (w, d), E(c), E(n)
        n = batch_size * num_context_ids
        pos_ids_flat, pos_mask_flat = pos_ids.view(n, -1), pos_mask.view(n, -1)
        neg_ids_flat, neg_mask_flat = neg_ids.view(n, -1), neg_mask.view(n, -1)

        joint_ids = torch.cat([center_ids, pos_ids_flat, neg_ids_flat], axis=0)
        joint_mask = torch.cat([center_mask, pos_mask_flat, neg_mask_flat], axis=0)

        center_sep_idx = 1 + 2
        other_sep_idx = num_metadata_samples + 2
        decoder_type_ids = torch.zeros([joint_ids.size()[0], max_decoder_len]).long().to(device)
        decoder_type_ids[:batch_size, -center_sep_idx:] = 1
        decoder_type_ids[batch_size:, -other_sep_idx:] = 1

        mu_joint, sigma_joint = self.decoder(
            input_ids=joint_ids, attention_mask=joint_mask, token_type_ids=decoder_type_ids)

        mu_center, sigma_center = mu_joint[:batch_size], sigma_joint[:batch_size]
        s = batch_size * (num_context_ids + 1)
        mu_pos_flat, sigma_pos_flat = mu_joint[batch_size:s], sigma_joint[batch_size:s]
        mu_neg_flat, sigma_neg_flat = mu_joint[s:], sigma_joint[s:]

        # Compute KL-divergence between center words and negative and reshape
        kl_pos_flat = compute_kl(mu_center_flat_q, sigma_center_flat_q, mu_pos_flat, sigma_pos_flat)
        kl_neg_flat = compute_kl(mu_center_flat_q, sigma_center_flat_q, mu_neg_flat, sigma_neg_flat)
        kl_pos = kl_pos_flat.view(batch_size, num_context_ids)
        kl_neg = kl_neg_flat.view(batch_size, num_context_ids)

        hinge_loss = (kl_pos - kl_neg + 1.0).clamp_min_(0)
        hinge_loss.masked_fill_(mask, 0)
        hinge_loss = hinge_loss.sum(1)

        recon_loss = compute_kl(mu_center_q, sigma_center_q, mu_center, sigma_center).squeeze(-1)
        return hinge_loss.mean(), recon_loss.mean()


class LMC(nn.Module):
    def __init__(self, args, token_vocab_size, metadata_vocab_size):
        super(LMC, self).__init__()
        self.encoder = LMCEncoder(token_vocab_size, metadata_vocab_size)
        self.decoder = LMCDecoder(token_vocab_size, metadata_vocab_size)

    def _compute_marginal(self, ids, metadata_ids):
        ids_tiled = ids.unsqueeze(-1).repeat(1, 1, metadata_ids.size()[-1])
        mu_marginal, sigma_marginal = self.decoder(ids_tiled, metadata_ids)
        return mu_marginal, sigma_marginal + 1e-3

    def forward(self, center_ids, center_metadata_ids, context_ids, context_metadata_ids, neg_ids, neg_metadata_ids,
                num_contexts, num_metadata_samples=None):
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size, num_context_ids = context_ids.size()
        mask_size = torch.Size([batch_size, num_context_ids])
        mask = mask_2D(mask_size, num_contexts).to(device)

        m_samples = context_metadata_ids.size()[-1]
        assert m_samples == num_metadata_samples

        # Compute center words
        mu_center_q, sigma_center_q, _ = self.encoder(center_ids, center_metadata_ids, context_ids, mask)
        mu_center_tiled_q = mu_center_q.unsqueeze(1).repeat(1, num_context_ids * m_samples, 1)
        sigma_center_tiled_q = sigma_center_q.unsqueeze(1).repeat(1, num_context_ids * m_samples, 1)
        mu_center_flat_q = mu_center_tiled_q.view(batch_size * num_context_ids * m_samples, -1)
        sigma_center_flat_q = sigma_center_tiled_q.view(batch_size * num_context_ids * m_samples, -1)

        # Compute decoded representations of (w, d), E(c), E(n)
        mu_center, sigma_center = self.decoder(center_ids, center_metadata_ids)
        mu_pos, sigma_pos = self._compute_marginal(context_ids, context_metadata_ids)
        mu_neg, sigma_neg = self._compute_marginal(neg_ids, neg_metadata_ids)

        # Flatten positive context
        mu_pos_flat = mu_pos.view(batch_size * num_context_ids * m_samples, -1)
        sigma_pos_flat = sigma_pos.view(batch_size * num_context_ids * m_samples, -1)

        # Flatten negative context
        mu_neg_flat = mu_neg.view(batch_size * num_context_ids * m_samples, -1)
        sigma_neg_flat = sigma_neg.view(batch_size * num_context_ids * m_samples, -1)

        # Compute KL-divergence between center words and negative and reshape
        kl_pos_flat = compute_kl(mu_center_flat_q, sigma_center_flat_q, mu_pos_flat, sigma_pos_flat)
        kl_neg_flat = compute_kl(mu_center_flat_q, sigma_center_flat_q, mu_neg_flat, sigma_neg_flat)
        kl_pos = kl_pos_flat.view(batch_size, num_context_ids, m_samples).mean(-1)
        kl_neg = kl_neg_flat.view(batch_size, num_context_ids, m_samples).mean(-1)

        hinge_loss = (kl_pos - kl_neg + 1.0).clamp_min_(0)
        hinge_loss.masked_fill_(mask, 0)
        hinge_loss = hinge_loss.sum(1)

        recon_loss = compute_kl(mu_center_q, sigma_center_q, mu_center, sigma_center).squeeze(-1)
        return hinge_loss.mean(), recon_loss.mean()
