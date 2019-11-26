import numpy as np
import torch
import torch.nn as nn

from model.encoder import Encoder

from model.utils import compute_kl


class VAE(nn.Module):
    def __init__(self, args, vocab_size ):
        super(VAE, self).__init__()

        self.encoder = Encoder(args, vocab_size)

        # The output representations of words(used in KL regularization and max_margin).
        self.embeddings_mu = nn.Embedding(vocab_size, args.latent_dim, padding_idx=0)
        self.embeddings_log_sigma = nn.Embedding(vocab_size, 1, padding_idx=0)
        log_weights_init = np.random.uniform(low=-3.5, high=-1.5, size=(vocab_size, 1))
        self.embeddings_log_sigma.weight.data = torch.FloatTensor(log_weights_init)

    def _max_margin(self, mu_q, sigma_q, pos_mu_p, pos_sigma_p, neg_mu_p, neg_sigma_p, margin=1.):
        """
        Computes a sum over context words margin(hinge loss).
        :param pos_context_words:  a tensor with true context words ids [batch_size x window_size]
        :param neg_context_words: a tensor with negative context words ids [batch_size x window_size]
        :param mask: a tensor binary mask where 0 indicates a padding
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

        kl_pos = compute_kl(mu_q_flat, sigma_q_flat, pos_mu_p_flat, pos_sigma_p_flat).reshape(batch_size, -1)
        kl_neg = compute_kl(mu_q_flat, sigma_q_flat, neg_mu_p_flat, neg_sigma_p_flat).view(batch_size, -1)

        hinge_loss = (kl_pos - kl_neg + margin).clamp_min_(0)
        return hinge_loss

    def forward(self, center_ids, context_ids, neg_context_ids=None):
        """
        :param center_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :return: TBD
        """
        mu_q, sigma_q = self.encoder(center_ids, context_ids)
        mu_p, sigma_p = self.embeddings_mu(center_ids), self.embeddings_log_sigma(center_ids).exp()

        pos_mu_p, pos_sigma_p = self.embeddings_mu(context_ids), self.embeddings_log_sigma(context_ids).exp()
        neg_mu_p, neg_sigma_p = self.embeddings_mu(neg_context_ids), self.embeddings_log_sigma(neg_context_ids).exp()

        kl = compute_kl(mu_q, sigma_q, mu_p, sigma_p).mean()
        max_margin = self._max_margin(mu_q, sigma_q, pos_mu_p, pos_sigma_p, neg_mu_p, neg_sigma_p).mean()
        return kl, max_margin


