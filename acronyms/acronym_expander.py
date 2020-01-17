import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_utils import compute_kl, mask_2D
from bsg_encoder import BSGEncoder, BSGEncoderLSTM


class AcronymExpander(nn.Module):
    def __init__(self, args, bsg_model, vocab):
        super(AcronymExpander, self).__init__()

        vocab_size = vocab.size()
        prev_vocab_size, embed_dim = bsg_model.embeddings_mu.weight.size()
        if args.random_priors:
            self.embeddings_mu = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=0)
            self.embeddings_log_sigma = nn.Embedding(vocab_size, embedding_dim=1, padding_idx=0)
            log_weights_init = np.random.uniform(low=-3.5, high=-1.5, size=(vocab_size, 1))
            self.embeddings_log_sigma.load_state_dict({'weight': torch.from_numpy(log_weights_init)})
        else:
            self.embeddings_mu = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=0)
            mu_init = np.random.normal(0, 1, size=(vocab_size, embed_dim))
            mu_init[:prev_vocab_size, :] = bsg_model.embeddings_mu.weight.detach().numpy()
            self.embeddings_mu.load_state_dict({'weight': torch.from_numpy(mu_init)})
            self.embeddings_log_sigma = nn.Embedding(vocab_size, embedding_dim=1, padding_idx=0)
            log_weights_init = np.random.uniform(low=-3.5, high=-1.5, size=(vocab_size, 1))
            log_weights_init[:prev_vocab_size, :] = bsg_model.embeddings_log_sigma.weight.detach().numpy()
            self.embeddings_log_sigma.load_state_dict({'weight': torch.from_numpy(log_weights_init)})

        if args.random_encoder:
            args.latent_dim, args.encoder_hidden_dim = bsg_model.encoder.u.weight.size()
            args.encoder_input_dim = args.encoder_hidden_dim
            if type(bsg_model.encoder) == BSGEncoderLSTM:
                self.encoder = BSGEncoderLSTM(args, vocab_size)
            else:
                self.encoder = BSGEncoder(args, vocab_size)
        else:
            self.encoder = bsg_model.encoder
            encoder_embed_dim = self.encoder.embeddings.weight.size()[-1]
            encoder_embed_init = np.random.normal(0, 1, size=(vocab_size, encoder_embed_dim))
            encoder_embed_init[:prev_vocab_size, :] = self.encoder.embeddings.weight.detach().numpy()
            self.encoder.embeddings = nn.Embedding(vocab_size, embedding_dim=encoder_embed_dim, padding_idx=0)
            self.encoder.embeddings.load_state_dict({'weight': torch.from_numpy(encoder_embed_init)})

    def _compute_priors(self, ids):
        return self.embeddings_mu(ids), self.embeddings_log_sigma(ids).exp()

    def encode_context(self, sf_ids, context_ids, global_ids, global_token_ct, use_att=False, att_style=None):
        batch_size, num_context_ids = context_ids.size()
        # First thing is to pass the SF with the context to the encoder
        mask = torch.BoolTensor(torch.Size([batch_size, num_context_ids]))
        mask.fill_(0)
        sf_mu, sf_sigma = self.encoder(sf_ids, context_ids, mask)
        top_global_weights = None
        if use_att:
            global_mu, global_sigma = self._compute_priors(global_ids)
            max_global_len = global_ids.size()[-1]
            global_size = torch.Size([batch_size, max_global_len])
            global_mask = mask_2D(global_size, global_token_ct)

            sf_mu_tiled_flat = sf_mu.unsqueeze(1).repeat(1, max_global_len, 1).view(batch_size *  max_global_len, -1)
            sf_sigma_tiled_flat = sf_sigma.unsqueeze(1).repeat(1, max_global_len, 1).view(batch_size * max_global_len, -1)

            global_kl = compute_kl(
                sf_mu_tiled_flat,
                sf_sigma_tiled_flat,
                global_mu.view(batch_size * max_global_len, -1),
                global_sigma.view(batch_size * max_global_len, -1)
            ).view(batch_size, max_global_len)

            global_kl.masked_fill_(global_mask, 1e5)
            global_att = nn.Softmax(-1)(-global_kl)
            k = 5
            top_global_weights = torch.topk(global_att, k=k).indices

            if att_style == 'weighted':
                global_coeff = global_att.unsqueeze(-1)
                weighted_global_mu = (global_mu * global_coeff).sum(1)
                weighted_global_sigma = (global_sigma * global_coeff).sum(1)
                sf_mu = weighted_global_mu
                sf_sigma = weighted_global_sigma
            elif att_style == 'two_pass':
                addl_context_ids = torch.zeros([batch_size, k]).long()
                for batch_idx in range(batch_size):
                    addl_context_ids[batch_idx, :] = global_ids[batch_idx, top_global_weights[batch_idx, :]]
                addl_context_ids = torch.cat([addl_context_ids, context_ids], axis=1)
                addl_mask = torch.BoolTensor(torch.Size([batch_size, k]))
                addl_mask.fill_(0)
                full_mask = torch.cat([addl_mask, mask], axis=1)
                sf_mu, sf_sigma = self.encoder(sf_ids, addl_context_ids, full_mask)
        return sf_mu, sf_sigma, top_global_weights

    def forward(self, sf_ids, metadata_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct, global_ids,
                global_token_ct, num_outputs, use_att=False, att_style='weighted'):
        """
        :param sf_ids: batch_size
        :param context_ids: batch_size, num_context_ids
        :param lf_ids: batch_size, max_output_size, max_lf_len
        :param lf_token_ct: batch_size, max_output_size - normalizer for lf_ids
        :param target_lf_ids: batch_size
        :param num_outputs: batch_size
        :param use_att: whether or not to use attention in the encoder with entire passage (global ids)
        :param att_style: to compute weighted average or re-compute encoder state with additional top k attended words
        :return:
        """
        batch_size, max_output_size, _ = lf_ids.size()

        # Next is to get prior representations for each LF in lf_ids
        lf_mu, lf_sigma = self._compute_priors(lf_ids)
        # Summarize LFs
        normalizer = lf_token_ct.unsqueeze(-1).clamp_min(1.0)
        lf_mu_sum, lf_sigma_sum = lf_mu.sum(-2) / normalizer, lf_sigma.sum(-2) / normalizer

        # Encode SFs in context (use attention or simply pass through Encoder)
        sf_mu, sf_sigma, top_global_weights = self.encode_context(sf_ids, context_ids, global_ids, global_token_ct,
                                                                  use_att=use_att, att_style=att_style)

        # Tile SFs across each LF and flatten both SFs and LFs
        sf_mu_flat = sf_mu.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        sf_sigma_flat = sf_sigma.unsqueeze(1).repeat(1, max_output_size, 1).view(batch_size * max_output_size, -1)
        lf_mu_flat = lf_mu_sum.view(batch_size * max_output_size, -1)
        lf_sigma_flat = lf_sigma_sum.view(batch_size * max_output_size, -1)

        output_dim = torch.Size([batch_size, max_output_size])
        output_mask = mask_2D(output_dim, num_outputs)

        kl = compute_kl(sf_mu_flat, sf_sigma_flat, lf_mu_flat, lf_sigma_flat).view(batch_size, max_output_size)
        score = -kl
        score.masked_fill_(output_mask, float('-inf'))
        return score, target_lf_ids, top_global_weights
