import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data


class Encoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, args.encoder_input_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.f = nn.Linear(args.encoder_input_dim * 2, args.encoder_hidden_dim, bias=True)
        self.u = nn.Linear(args.encoder_hidden_dim, args.latent_dim, bias=True)
        self.v = nn.Linear(args.encoder_hidden_dim, 1, bias=True)
        
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
        merged_embeds = self.dropout(torch.cat([center_embedding_tiled, context_embedding], dim=-1))

        h_reps = self.dropout(F.relu(self.f(merged_embeds)))
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, h_reps.size()[-1])
        h_reps.masked_fill_(mask_tiled, 0)
        h = h_reps.sum(1)
        return self.u(h), self.v(h).exp()


class EncoderLSTM(nn.Module):
    """
    Unused for now
    """
    def __init__(self, args, vocab_size):
        super(EncoderLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, args.encoder_input_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            args.encoder_input_dim * 2, args.encoder_hidden_dim, bidirectional=True, batch_first=True)
        self.att = nn.Linear(args.encoder_hidden_dim * 2, 1, bias=True)
        self.u = nn.Linear(args.encoder_hidden_dim * 2, args.latent_dim, bias=True)
        self.v = nn.Linear(args.encoder_hidden_dim * 2, 1, bias=True)
        self.softmax = nn.Softmax(1)

    def forward(self, center_ids, context_ids, mask):
        """
        :param center_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), logvar (batch_size, 1)
        """
        batch_size, num_context_ids = context_ids.shape

        center_embedding = self.embeddings(center_ids)
        context_embedding = self.embeddings(context_ids)

        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = self.dropout(torch.cat([center_embedding_tiled, context_embedding], dim=-1))
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, merged_embeds.size()[-1])
        merged_embeds.masked_fill_(mask_tiled, 0)

        h, _ = self.lstm(merged_embeds)
        att_scores = self.att(h).squeeze(-1)
        att_scores.masked_fill_(mask, -1e5)
        att_weights = self.softmax(att_scores)
        h_sum = (att_weights.unsqueeze(-1) * h).sum(1)
        return self.u(h_sum), self.v(h_sum).exp()
