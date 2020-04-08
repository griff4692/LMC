import os
import sys

import numpy as np
import torch
from torch import nn
import torch.utils.data
from transformers import AlbertConfig, AlbertModel

from albert_encode import encode


class LMCEncoderBERT(nn.Module):
    def __init__(self, args, token_vocab_size, output_dim=100):
        super(LMCEncoderBERT, self).__init__()
        self.pool_layers = args.pool_bert

        if args.debug_model:
            bert_dim = 100
            num_hidden_layers = 1
            embedding_size = 100
            intermediate_size = 100
            output_dim = 100
        else:
            bert_dim = 512
            num_hidden_layers = 4
            embedding_size = 128
            intermediate_size = 512
        num_attention_heads = max(1, bert_dim // 64)
        print('Using {} attention heads in encoder'.format(num_attention_heads))

        config = AlbertConfig(
            vocab_size=token_vocab_size,
            embedding_size=embedding_size,
            hidden_size=bert_dim,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,  # 3072 is default
            num_attention_heads=num_attention_heads,
            output_hidden_states=self.pool_layers
        )

        self.bert = AlbertModel(config)
        self.u = nn.Linear(bert_dim, output_dim, bias=True)
        self.v = nn.Linear(bert_dim, 1, bias=True)
        self.att_linear = nn.Linear(bert_dim, 1, bias=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, **kwargs):
        h = self.dropout(encode(self, **kwargs))
        return self.u(h), self.v(h).exp(), None


class LMCEncoder(nn.Module):
    def __init__(self, token_vocab_size, metadata_vocab_size, input_dim=128, hidden_dim=64, output_dim=100):
        super(LMCEncoder, self).__init__()
        self.token_embeddings = nn.Embedding(token_vocab_size, input_dim, padding_idx=0)
        self.metadata_embeddings = nn.Embedding(metadata_vocab_size, hidden_dim * 2, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_dim * 3, hidden_dim, bidirectional=True, batch_first=True)
        self.combine_att = nn.Linear(hidden_dim * 4, 2, bias=True)
        self.u = nn.Linear(hidden_dim * 2, output_dim, bias=True)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=True)

        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

    def forward(self, center_ids, metadata_ids, context_ids, mask, context_mask_p=0.2):
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

        context_embedding = self.token_embeddings(context_ids)
        if context_mask_p is not None:
            context_mask = torch.FloatTensor(batch_size, num_context_ids).uniform_().to(device) < context_mask_p
            mask.masked_fill_(context_mask, True)

        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        metadata_embedding_tiled = metadata_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = torch.cat([center_embedding_tiled, context_embedding, metadata_embedding_tiled], dim=-1)
        merged_embeds = self.dropout(merged_embeds)
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, merged_embeds.size()[-1])
        merged_embeds.masked_fill_(mask_tiled, 0)

        h_reps, (h_sum, _) = self.lstm(merged_embeds)

        e_dim = h_reps.size()[-1]
        att_scores = torch.bmm(h_reps, metadata_embedding.unsqueeze(-1)).squeeze(-1) / np.sqrt(e_dim)
        att_scores.masked_fill_(mask, -1e5)
        att_weights = self.softmax(att_scores)
        h_sum = (att_weights.unsqueeze(-1) * h_reps).sum(1)

        return self.u(h_sum), self.v(h_sum).exp()
