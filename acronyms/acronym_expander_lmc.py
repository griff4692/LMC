import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/lmc_context/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_utils import compute_kl, mask_2D


class AcronymExpanderLMC(nn.Module):
    def __init__(self, args, lmc_model, token_vocab):
        super(AcronymExpanderLMC, self).__init__()

        token_vocab_size = token_vocab.size()
        prev_token_vocab_size, encoder_embed_dim = lmc_model.encoder.token_embeddings.weight.size()

        prev_encoder_token_embeddings = lmc_model.encoder.token_embeddings.weight.detach().numpy()
        self.encoder = lmc_model.encoder
        self.metadata = args.metadata
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

    def _compute_marginal(self, ids, metadata_ids, metadata_p, normalizer=None):
        batch_size, max_output, num_metadata = metadata_p.size()
        max_lf_ngram = ids.size()[-1]
        normalizer_tiled_flat = normalizer.unsqueeze(2).repeat(1, 1, num_metadata, 1).view(
            batch_size * max_output * num_metadata, 1)
        ids_tiled = ids.unsqueeze(2).repeat(1, 1, num_metadata, 1)
        ids_tiled_flat = ids_tiled.view(batch_size * max_output * num_metadata, max_lf_ngram)
        metadata_ids_flat = metadata_ids.reshape(-1)
        mu_marginal_flat, sigma_marginal_flat = self.decoder(
            ids_tiled_flat, metadata_ids_flat, normalizer=normalizer_tiled_flat)
        mu_marginal = mu_marginal_flat.view(batch_size, max_output, num_metadata, -1)
        sigma_marginal = sigma_marginal_flat.view(batch_size, max_output, num_metadata, -1)
        metadata_p = metadata_p.unsqueeze(-1)
        mu = (metadata_p * mu_marginal).sum(2)
        sigma = (metadata_p * sigma_marginal).sum(2)
        return mu, sigma + 1e-3

    def forward(self, sf_ids, section_ids, category_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct,
                lf_metadata_ids, lf_metadata_p, num_outputs, num_contexts):
        batch_size, max_output_size, max_lf_ngram = lf_ids.size()
        _, num_context_ids = context_ids.size()

        # Compute SF contexts
        # Mask padded context ids
        mask_size = torch.Size([batch_size, num_context_ids])
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = mask_2D(mask_size, num_contexts).to(device_str)
        sf_mu, sf_sigma, _ = self.encoder(
            sf_ids, section_ids, context_ids, mask, center_mask_p=None, context_mask_p=None)

        # Tile SFs across each LF and flatten both SFs and LFs
        sf_mu_flat = sf_mu.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        sf_sigma_flat = sf_sigma.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)

        # Compute E[LF]
        normalizer = lf_token_ct.unsqueeze(-1).clamp_min(1.0)
        lf_mu, lf_sigma = self._compute_marginal(lf_ids, lf_metadata_ids, lf_metadata_p, normalizer=normalizer)
        lf_mu_flat = lf_mu.view(batch_size * max_output_size, -1)
        lf_sigma_flat = lf_sigma.view(batch_size * max_output_size, 1)

        output_dim = torch.Size([batch_size, max_output_size])
        output_mask = mask_2D(output_dim, num_outputs).to(device_str)

        kl = compute_kl(sf_mu_flat, sf_sigma_flat, lf_mu_flat, lf_sigma_flat).view(batch_size, max_output_size)
        score = -kl
        score.masked_fill_(output_mask, float('-inf'))

        return score, target_lf_ids
