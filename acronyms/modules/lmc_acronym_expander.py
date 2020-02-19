import os
import sys

import numpy as np
import torch
import torch.nn as nn

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from compute_utils import compute_kl, mask_2D


class LMCAcronymExpander(nn.Module):
    """
    Loads in a pre-trained LMC Model and ranks candidate Long Forms for each Acronym Short Form
    Ranking function for each LF_k: DKL(q(z|SF,d,c)||∑dp(z|LFk,d)βd|LF_k)
    q = variational distribution
    z = latent meaning cell
    d = metadata

    Please refer to LMC paper for details on ranking function and derivations.
    """
    def __init__(self, lmc_model, token_vocab,):
        super(LMCAcronymExpander, self).__init__()

        token_vocab_size = token_vocab.size()
        prev_token_vocab_size, encoder_embed_dim = lmc_model.encoder.token_embeddings.weight.size()

        # In main evaluation script, we add long forms not present in vocabulary so we must expand embedding dimensions
        # Merge weights from pre-trained model to encoder and randomly initialize extra vocabulary items added
        prev_encoder_token_embeddings = lmc_model.encoder.token_embeddings.weight.detach().numpy()
        self.encoder = lmc_model.encoder
        self.encoder.token_embeddings = nn.Embedding(token_vocab_size, encoder_embed_dim, padding_idx=0)
        encoder_init = np.random.normal(0, 1, size=(token_vocab_size, encoder_embed_dim))
        encoder_init[:prev_token_vocab_size, :] = prev_encoder_token_embeddings
        self.encoder.token_embeddings.load_state_dict({'weight': torch.from_numpy(encoder_init)})

        # Merge weights from pre-trained model to decoder and randomly initialize extra vocabulary items added
        prev_token_vocab_size_decoder, decoder_embed_dim = lmc_model.decoder.token_embeddings.weight.size()
        assert prev_token_vocab_size == prev_token_vocab_size_decoder
        prev_decoder_token_embeddings = lmc_model.decoder.token_embeddings.weight.detach().numpy()
        self.decoder = lmc_model.decoder
        self.decoder.token_embeddings = nn.Embedding(token_vocab_size, decoder_embed_dim, padding_idx=0)
        decoder_init = np.random.normal(0, 1, size=(token_vocab_size, decoder_embed_dim))
        decoder_init[:prev_token_vocab_size, :] = prev_decoder_token_embeddings
        self.decoder.token_embeddings.load_state_dict({'weight': torch.from_numpy(decoder_init)})

    def _compute_marginal(self, ids, metadata_ids, normalizer=None):
        """
        :param ids: LongTensor of batch_size x max_output_size x max_lf_len
        :param metadata_ids: batch_size x max_output_size x max_num_metadata
        :param normalizer: batch_size x max_output_size x 1.  Number of n-grams in each LF (used for computing mean)
        :return: Gaussian parameters mu (batch_size x max_output_size )
        """
        batch_size, max_output, num_metadata = metadata_ids.size()
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
        return mu_marginal, sigma_marginal

    def forward(self, sf_ids, section_ids, category_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct,
                lf_metadata_ids, lf_metadata_p, num_outputs, num_contexts):
        """
        :param sf_ids: LongTensor of batch_size
        :param section_ids: LongTensor of batch_size
        :param category_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param lf_ids: LongTensor of batch_size x max_output_size x max_lf_len
        :param target_lf_ids: LongTensor of batch_size representing which index in lf_ids lies the target LF
        :param lf_token_ct: LongTensor of batch_size x max_output_size.  N-gram count for each LF (used for masking)
        :param lf_metadata_ids: batch_size x max_output_size x max_num_metadata. Ids for every metadata LF appears in
        :param lf_metadata_p: batch_size x max_output_size x max_num_metadata
        Empirical probability for lf_metadata_ids ~ p(metadata|LF)
        :param num_outputs: list representing the number of target LFs for each row in batch.
        Used for masking to avoid returning invalid predictions.
        :param num_contexts: LongTensor of batch_size.  The actual window size of the SF context.
        Many are shorter than target of 2 * context_window.
        :return: scores for each candidate LF, target_lf_ids, rel_weights (output of encoder gating function)
        """
        batch_size, max_output_size, max_lf_ngram = lf_ids.size()
        _, num_context_ids = context_ids.size()

        # Compute SF contexts
        # Mask padded context ids
        mask_size = torch.Size([batch_size, num_context_ids])
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = mask_2D(mask_size, num_contexts).to(device_str)
        sf_mu, sf_sigma, rel_weights = self.encoder(
            sf_ids, section_ids, context_ids, mask, center_mask_p=None, context_mask_p=None)

        num_metadata = lf_metadata_ids.size()[-1]

        # Tile SFs across each LF and flatten both SFs and LFs
        sf_mu_flat = sf_mu.unsqueeze(1).repeat(1, max_output_size * num_metadata, 1).view(
            batch_size * max_output_size * num_metadata, -1)
        sf_sigma_flat = sf_sigma.unsqueeze(1).repeat(1, max_output_size * num_metadata, 1).view(
            batch_size * max_output_size * num_metadata, -1)

        # Compute E[LF]
        normalizer = lf_token_ct.unsqueeze(-1).clamp_min(1.0)
        lf_mu, lf_sigma = self._compute_marginal(lf_ids, lf_metadata_ids, normalizer=normalizer)
        lf_mu_flat = lf_mu.view(batch_size * max_output_size * num_metadata, -1)
        lf_sigma_flat = lf_sigma.view(batch_size * max_output_size * num_metadata, 1)
        output_dim = torch.Size([batch_size, max_output_size])
        output_mask = mask_2D(output_dim, num_outputs).to(device_str)

        kl_marginal = compute_kl(sf_mu_flat, sf_sigma_flat, lf_mu_flat, lf_sigma_flat).view(
            batch_size, max_output_size, num_metadata)
        kl = (kl_marginal * lf_metadata_p).sum(2)
        score = -kl

        score.masked_fill_(output_mask, float('-inf'))
        return score, target_lf_ids, rel_weights
