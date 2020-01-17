import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/lmc_context/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_utils import compute_kl, mask_2D
from lmc_context_encoder import LMCContextEncoder
from lmc_context_decoder import LMCDecoder


class AcronymExpanderLMC(nn.Module):
    def __init__(self, args, lmc_model, token_vocab):
        super(AcronymExpanderLMC, self).__init__()

        token_vocab_size = token_vocab.size()
        prev_token_vocab_size, encoder_embed_dim = lmc_model.encoder.token_embeddings.weight.size()

        prev_encoder_token_embeddings = lmc_model.encoder.token_embeddings.weight.detach().numpy()
        self.encoder = lmc_model.encoder
        self.encoder.token_embeddings = nn.Embedding(token_vocab_size, encoder_embed_dim, padding_idx=0)
        encoder_init = np.random.normal(0, 1, size=(token_vocab_size, encoder_embed_dim))
        encoder_init[:prev_token_vocab_size, :] = prev_encoder_token_embeddings
        self.encoder.token_embeddings.load_state_dict({'weight': torch.from_numpy(encoder_init)})

        prev_token_vocab_size_decoder, decoder_embed_dim = lmc_model.decoder.token_embeddings.weight.size()
        assert prev_token_vocab_size == prev_token_vocab_size_decoder
        prev_decoder_token_embeddings = lmc_model.decoder.token_embeddings.weight.detach().numpy()
        self.decoder = lmc_model.decoder
        self.decoder.token_embeddings = nn.Embedding(token_vocab_size, decoder_embed_dim, padding_idx=0)
        decoder_init = np.random.normal(0, 1, size=(token_vocab_size, decoder_embed_dim))
        decoder_init[:prev_token_vocab_size, :] = prev_decoder_token_embeddings
        self.decoder.token_embeddings.load_state_dict({'weight': torch.from_numpy(decoder_init)})

    def _compute_marginal(self, ids, metadata_p, normalizer=None):
        batch_size, max_output, num_metadata = metadata_p.size()
        max_lf_ngram = ids.size()[-1]
        normalizer_tiled_flat = normalizer.unsqueeze(2).repeat(1, 1, num_metadata, 1).view(
            batch_size * max_output * num_metadata, 1)
        ids_tiled = ids.unsqueeze(2).repeat(1, 1, num_metadata, 1)
        ids_tiled_flat = ids_tiled.view(batch_size * max_output * num_metadata, max_lf_ngram)
        metadata_ids_flat = torch.arange(num_metadata).unsqueeze(0).repeat(batch_size * max_output, 1).view(-1)
        mu_marginal_flat, sigma_marginal_flat = self.decoder(
            ids_tiled_flat, metadata_ids_flat, normalizer=normalizer_tiled_flat)
        mu_marginal = mu_marginal_flat.view(batch_size, max_output, num_metadata, -1)
        sigma_marginal = sigma_marginal_flat.view(batch_size, max_output, num_metadata, -1)
        metadata_p = metadata_p.unsqueeze(-1)
        mu = (metadata_p * mu_marginal).sum(2)
        sigma = (metadata_p * sigma_marginal).sum(2)
        return mu, sigma + 1e-3

    def forward(self, sf_ids, metadata_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct, global_ids,
                global_token_ct, lf_metadata_p, num_outputs, use_att=False, att_style=None, compute_marginal=False):
        batch_size, max_output_size, max_lf_ngram = lf_ids.size()
        _, num_context_ids = context_ids.size()

        # Compute SF contexts
        # TODO actually mask padded context ids
        mask = torch.BoolTensor(torch.Size([batch_size, num_context_ids]))
        mask.fill_(0)
        sf_mu, sf_sigma = self.encoder(sf_ids, metadata_ids, context_ids, mask)

        # Tile SFs across each LF and flatten both SFs and LFs
        sf_mu_flat = sf_mu.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        sf_sigma_flat = sf_sigma.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)

        # Compute E[LF]
        normalizer = lf_token_ct.unsqueeze(-1).clamp_min(1.0)
        if compute_marginal:
            lf_mu, lf_sigma = self._compute_marginal(lf_ids, lf_metadata_p, normalizer=normalizer)
            lf_mu_flat = lf_mu.view(batch_size * max_output_size, -1)
            lf_sigma_flat = lf_sigma.view(batch_size * max_output_size, 1)
        else:
            normalizer_flat = normalizer.view(batch_size * max_output_size, 1)
            metadata_ids_tiled = metadata_ids.unsqueeze(1).repeat(1, max_output_size)
            metadata_ids_tiled_flat = metadata_ids_tiled.view(batch_size * max_output_size)
            lf_ids_flat = lf_ids.view(batch_size * max_output_size, max_lf_ngram)
            lf_mu_flat, lf_sigma_flat = self.decoder(lf_ids_flat, metadata_ids_tiled_flat, normalizer=normalizer_flat)

        output_dim = torch.Size([batch_size, max_output_size])
        output_mask = mask_2D(output_dim, num_outputs)

        kl = compute_kl(sf_mu_flat, sf_sigma_flat, lf_mu_flat, lf_sigma_flat).view(batch_size, max_output_size)
        score = -kl
        score.masked_fill_(output_mask, float('-inf'))

        return score, target_lf_ids, None
