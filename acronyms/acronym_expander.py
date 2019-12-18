import torch
import torch.nn as nn

from compute_utils import compute_kl, mask_2D


class AcronymExpander(nn.Module):
    def __init__(self, bsg_model):
        super(AcronymExpander, self).__init__()
        self.embeddings_mu = bsg_model.embeddings_mu
        self.embeddings_log_sigma = bsg_model.embeddings_log_sigma
        self.encoder = bsg_model.encoder
        self.softmax = nn.Softmax(dim=-1)

    def _compute_priors(self, ids):
        return self.embeddings_mu(ids), self.embeddings_log_sigma(ids).exp()

    def forward(self, sf_ids, context_ids, lf_ids, target_lf_ids, num_outputs):
        """
        :param sf_ids: batch_size
        :param context_ids: batch_size, num_context_ids
        :param lf_ids: batch_size, max_output_size, max_lf_len
        :param target_lf_ids: batch_size
        :param num_outputs: batch_size
        :return:
        """
        batch_size, num_context_ids = context_ids.size()
        max_output_size = lf_ids.size()[1]

        output_dim = torch.Size([batch_size, max_output_size])
        output_mask = mask_2D(output_dim, num_outputs)

        # First thing is to pass the SF with the context to the encoder
        mask = torch.BoolTensor(torch.Size([batch_size, num_context_ids]))
        mask.fill_(0)
        sf_mu, sf_sigma = self.encoder(sf_ids, context_ids, mask)

        # Next is to get prior representations for each LF in lf_ids
        lf_mu, lf_sigma = self._compute_priors(lf_ids)

        # Summarize LFs
        lf_mu_sum, lf_sigma_sum = lf_mu.mean(-2), lf_sigma.mean(-2)

        # Tile SFs across each LF and flatten both SFs and LFs
        sf_mu_flat = sf_mu.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        sf_sigma_flat = sf_sigma.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        lf_mu_flat = lf_mu_sum.view(batch_size * max_output_size, -1)
        lf_sigma_flat = lf_sigma_sum.view(batch_size * max_output_size, -1)

        kl = compute_kl(sf_mu_flat, sf_sigma_flat, lf_mu_flat, lf_sigma_flat).view(batch_size, max_output_size)

        kl.masked_fill_(output_mask, float('-inf'))
        return kl, target_lf_ids
