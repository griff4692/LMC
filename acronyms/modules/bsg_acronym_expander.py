import os
import sys

import numpy as np
import torch
import torch.nn as nn

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from compute_utils import compute_kl, mask_2D


class BSGAcronymExpander(nn.Module):
    """
    Loads in a pre-trained BSG Model and ranks candidate Long Forms for each Acronym Short Form
    Ranking function for each LF_k: DKL(q(z|SF,c)||p(z|LFk))
    """
    def __init__(self, bsg_model, token_vocab):
        super(BSGAcronymExpander, self).__init__()

        vocab_size = token_vocab.size()
        prev_vocab_size, embed_dim = bsg_model.embeddings_mu.weight.size()

        # In main evaluation script, we add long forms not present in vocabulary so we must expand embedding dimensions
        # Merge weights from pre-trained model to decoder and randomly initialize extra vocabulary items added
        self.embeddings_mu = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=0)
        mu_init = np.random.normal(0, 1, size=(vocab_size, embed_dim))
        mu_init[:prev_vocab_size, :] = bsg_model.embeddings_mu.weight.detach().numpy()
        self.embeddings_mu.load_state_dict({'weight': torch.from_numpy(mu_init)})
        self.embeddings_log_sigma = nn.Embedding(vocab_size, embedding_dim=1, padding_idx=0)
        log_weights_init = np.random.uniform(low=-3.5, high=-1.5, size=(vocab_size, 1))
        log_weights_init[:prev_vocab_size, :] = bsg_model.embeddings_log_sigma.weight.detach().numpy()
        self.embeddings_log_sigma.load_state_dict({'weight': torch.from_numpy(log_weights_init)})

        # Merge weights from pre-trained model to encoder and randomly initialize extra vocabulary items added
        self.encoder = bsg_model.encoder
        encoder_embed_dim = self.encoder.embeddings.weight.size()[-1]
        encoder_embed_init = np.random.normal(0, 1, size=(vocab_size, encoder_embed_dim))
        encoder_embed_init[:prev_vocab_size, :] = self.encoder.embeddings.weight.detach().numpy()
        self.encoder.embeddings = nn.Embedding(vocab_size, embedding_dim=encoder_embed_dim, padding_idx=0)
        self.encoder.embeddings.load_state_dict({'weight': torch.from_numpy(encoder_embed_init)})

    def _compute_priors(self, ids):
        """
        :param ids:
        :return:
        """
        return self.embeddings_mu(ids), self.embeddings_log_sigma(ids).exp()

    def encode_context(self, sf_ids, context_ids, num_contexts):
        batch_size, num_context_ids = context_ids.size()

        # Mask padded context ids
        mask_size = torch.Size([batch_size, num_context_ids])
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = mask_2D(mask_size, num_contexts).to(device_str)

        sf_mu, sf_sigma = self.encoder(sf_ids, context_ids, mask, token_mask_p=None)
        return sf_mu, sf_sigma

    def forward(self, sf_ids, section_ids, category_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct,
                lf_metadata_ids, num_outputs, num_contexts):
        """
        :param sf_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param lf_ids: LongTensor of batch_size x max_output_size x max_lf_len
        :param lf_token_ct: batch_size, max_output_size - normalizer for lf_ids
        :param target_lf_ids: LongTensor of batch_size representing which index in lf_ids lies the target LF
        :param num_outputs: list representing the number of target LFs for each row in batch.
        :return:
        """
        batch_size, max_output_size, _ = lf_ids.size()

        # Next is to get prior representations for each LF in lf_ids
        lf_mu, lf_sigma = self._compute_priors(lf_ids)
        # Summarize LFs
        normalizer = lf_token_ct.unsqueeze(-1).clamp_min(1.0)
        lf_mu_sum, lf_sigma_sum = lf_mu.sum(-2) / normalizer, lf_sigma.sum(-2) / normalizer

        combined_mu = []
        combined_sigma = []

        # Encode SFs in context
        sf_mu_tokens, sf_sigma_tokens = self.encode_context(sf_ids, context_ids, num_contexts)
        combined_mu.append(sf_mu_tokens)
        combined_sigma.append(sf_sigma_tokens)

        # For MBSGE ensemble method, we leverage section ids and note category ids
        if len(section_ids.nonzero()) > 0:
            sf_mu_sec, sf_sigma_sec = self.encode_context(section_ids, context_ids, num_contexts)
            combined_mu.append(sf_mu_sec)
            combined_sigma.append(sf_sigma_sec)

        if len(category_ids.nonzero()) > 0:
            sf_mu_cat, sf_sigma_cat = self.encode_context(category_ids, context_ids, num_contexts)
            combined_mu.append(sf_mu_cat)
            combined_sigma.append(sf_sigma_cat)

        combined_mu = torch.cat(list(map(lambda x: x.unsqueeze(1), combined_mu)), axis=1)
        combined_sigma = torch.cat(list(map(lambda x: x.unsqueeze(1), combined_sigma)), axis=1)

        sf_mu = combined_mu.mean(1)
        sf_sigma = combined_sigma.mean(1)

        # Tile SFs across each LF and flatten both SFs and LFs
        sf_mu_flat = sf_mu.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        sf_sigma_flat = sf_sigma.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        lf_mu_flat = lf_mu_sum.view(batch_size * max_output_size, -1)
        lf_sigma_flat = lf_sigma_sum.view(batch_size * max_output_size, -1)

        output_dim = torch.Size([batch_size, max_output_size])
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_mask = mask_2D(output_dim, num_outputs).to(device_str)

        kl = compute_kl(sf_mu_flat, sf_sigma_flat, lf_mu_flat, lf_sigma_flat).view(batch_size, max_output_size)
        score = -kl
        score.masked_fill_(output_mask, float('-inf'))
        return score, target_lf_ids, None
