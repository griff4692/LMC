import sys

import numpy as np
import torch
from torch import nn
import torch.utils.data

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_utils import compute_att


class LMCContextEncoder(nn.Module):
    def __init__(self, token_vocab_size, metadata_vocab_size, input_dim=100, hidden_dim=64, output_dim=100):
        super(LMCContextEncoder, self).__init__()
        self.token_embeddings = nn.Embedding(token_vocab_size, input_dim, padding_idx=0)
        self.metadata_embeddings = nn.Embedding(metadata_vocab_size, hidden_dim * 2, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.combine_att = nn.Linear(hidden_dim * 4, 2, bias=True)
        self.u = nn.Linear(hidden_dim * 2, output_dim, bias=True)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=True)

        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

    def _mask_embeddings(self, e, mask_p, device='gpu'):
        if mask_p is None:
            return e
        e_size = e.size()
        mask = (torch.FloatTensor(e_size[:-1]).uniform_().to(device) < mask_p).unsqueeze(-1)
        mask = mask.repeat(1, e_size[-1]) if len(e_size) == 2 else mask.repeat(1, 1, e_size[-1])
        e.masked_fill_(mask, 0.0)

    def forward(self, center_ids, metadata_ids, context_ids, mask, center_mask_p=0.2, context_mask_p=0.2,
                metadata_mask_p=None):
        """
        :param center_ids: LongTensor of batch_size
        :param metadata_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), var (batch_size, 1)
        """
        batch_size, num_context_ids = context_ids.shape
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lstm.flatten_parameters()

        center_embedding = self.token_embeddings(center_ids)
        metadata_embedding = self.metadata_embeddings(metadata_ids)
        self._mask_embeddings(center_embedding, center_mask_p, device=device)

        context_embedding = self.token_embeddings(context_ids)
        if context_mask_p is not None:
            context_mask = torch.FloatTensor(batch_size, num_context_ids).uniform_().to(device) < context_mask_p
            mask.masked_fill_(context_mask, True)

        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = torch.cat([center_embedding_tiled, context_embedding], dim=-1)
        merged_embeds = self.dropout(merged_embeds)
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, merged_embeds.size()[-1])
        merged_embeds.masked_fill_(mask_tiled, 0)

        h_reps, (h_sum, _) = self.lstm(merged_embeds)
        # h_sum = h_sum.transpose(1, 0).contiguous().view(batch_size, -1)

        e_dim = h_reps.size()[-1]
        att_scores = torch.bmm(h_reps, metadata_embedding.unsqueeze(-1)).squeeze(-1) / np.sqrt(e_dim)
        att_scores.masked_fill_(mask, -1e5)
        att_weights = self.softmax(att_scores)
        h_sum = (att_weights.unsqueeze(-1) * h_reps).sum(1)

        rel_weights = self.softmax(
            self.tanh(self.combine_att(self.dropout(torch.cat([h_sum, metadata_embedding], axis=1)))))

        met_weight = rel_weights[:, 0].unsqueeze(1)
        token_weight = rel_weights[:, 1].unsqueeze(1)
        summary = self.dropout((met_weight * metadata_embedding + token_weight * h_sum))

        return self.u(summary), self.v(summary).exp()


class LMCContextEncoderOld(nn.Module):
    def __init__(self, token_vocab_size, metadata_vocab_size, input_dim=64, hidden_dim=64, output_dim=100):
        super(LMCContextEncoderOld, self).__init__()
        self.token_embeddings = nn.Embedding(token_vocab_size, input_dim, padding_idx=0)
        self.metadata_embeddings = nn.Embedding(metadata_vocab_size, input_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_dim * 3, hidden_dim, bidirectional=True, batch_first=True)
        self.att = nn.Linear(hidden_dim * 2, 1, bias=True)
        self.u = nn.Linear(hidden_dim * 2, output_dim, bias=True)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=True)

    def _mask_embeddings(self, e, mask_p, device='gpu'):
        if mask_p is None:
            return e
        e_size = e.size()
        mask = (torch.FloatTensor(e_size[:-1]).uniform_().to(device) < mask_p).unsqueeze(-1)
        mask = mask.repeat(1, e_size[-1]) if len(e_size) == 2 else mask.repeat(1, 1, e_size[-1])
        e.masked_fill_(mask, 0.0)

    def forward(self, center_ids, metadata_ids, context_ids, mask, center_mask_p=0.2, metadata_mask_p=0.2,
                context_mask_p=0.2):
        """
        :param center_ids: LongTensor of batch_size
        :param metadata_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), var (batch_size, 1)
        """
        batch_size, num_context_ids = context_ids.shape
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lstm.flatten_parameters()

        center_embedding = self.token_embeddings(center_ids)
        metadata_embedding = self.metadata_embeddings(metadata_ids)
        self._mask_embeddings(center_embedding, center_mask_p, device=device)
        self._mask_embeddings(metadata_embedding, metadata_mask_p, device=device)

        context_embedding = self.token_embeddings(context_ids)
        if context_mask_p is not None:
            context_mask = torch.FloatTensor(batch_size, num_context_ids).uniform_().to(device) < context_mask_p
            mask.masked_fill_(context_mask, True)

        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        metadata_embedding_tiled = metadata_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = torch.cat([center_embedding_tiled, metadata_embedding_tiled, context_embedding], dim=-1)
        merged_embeds = self.dropout(merged_embeds)
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, merged_embeds.size()[-1])
        merged_embeds.masked_fill_(mask_tiled, 0)

        h_reps, (h, _) = self.lstm(merged_embeds)
        h_sum = self.dropout(compute_att(h_reps, mask, self.att))
        return self.u(h_sum), self.v(h_sum).exp()
