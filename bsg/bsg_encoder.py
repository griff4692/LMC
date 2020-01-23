import sys

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_utils import compute_att


class BSGEncoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(BSGEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, args.input_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.f = nn.Linear(args.input_dim * 2, args.hidden_dim, bias=True)
        self.att = nn.Linear(args.hidden_dim, 1, bias=True)
        self.u = nn.Linear(args.hidden_dim, args.input_dim, bias=True)
        self.v = nn.Linear(args.hidden_dim, 1, bias=True)

    def forward(self, center_ids, context_ids, mask):
        """
        :param center_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), logvar (batch_size, 1)
        """
        center_embedding = self.embeddings(center_ids)
        context_embedding = self.embeddings(context_ids)

        num_context_ids = context_embedding.shape[1]
        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = torch.cat([center_embedding_tiled, context_embedding], dim=-1)
        merged_embeds = self.dropout(merged_embeds)

        h_reps = self.dropout(F.relu(self.f(merged_embeds)))
        h = compute_att(h_reps, mask, self.att)
        return self.u(h), self.v(h).exp()


class BSGEncoderLSTM(nn.Module):
    def __init__(self, args, vocab_size):
        super(BSGEncoderLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, args.input_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(args.input_dim * 2, args.hidden_dim, bidirectional=True, batch_first=True)
        self.att = nn.Linear(args.hidden_dim * 2, 1, bias=True)
        self.u = nn.Linear(args.hidden_dim * 2, args.input_dim, bias=True)
        self.v = nn.Linear(args.hidden_dim * 2, 1, bias=True)

    def forward(self, center_ids, context_ids, mask, token_mask_p=0.2):
        """
        :param center_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), var (batch_size, 1)
        """
        batch_size, num_context_ids = context_ids.shape

        center_embedding = self.embeddings(center_ids)

        if token_mask_p is not None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            context_mask = torch.FloatTensor(batch_size, num_context_ids).uniform_().to(device) < token_mask_p
            context_ids.masked_fill_(context_mask, 0)
        context_embedding = self.embeddings(context_ids)

        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = torch.cat([center_embedding_tiled, context_embedding], dim=-1)
        merged_embeds = self.dropout(merged_embeds)
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, merged_embeds.size()[-1])
        merged_embeds.masked_fill_(mask_tiled, 0)

        h_reps, (h, _) = self.lstm(merged_embeds)
        h_sum = self.dropout(compute_att(h_reps, mask, self.att))
        return self.u(h_sum), self.v(h_sum).exp()