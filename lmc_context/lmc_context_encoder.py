import torch
from torch import nn
import torch.utils.data


class LMCContextEncoder(nn.Module):
    def __init__(self, args, token_vocab_size, metadata_vocab_size):
        super(LMCContextEncoder, self).__init__()
        self.token_embeddings = nn.Embedding(token_vocab_size, args.input_dim, padding_idx=0)
        self.metadata_embeddings = nn.Embedding(metadata_vocab_size, args.input_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(args.input_dim * 3, args.hidden_dim, bidirectional=True, batch_first=True)
        self.u = nn.Linear(args.hidden_dim * 2, args.latent_dim, bias=True)
        self.v = nn.Linear(args.hidden_dim * 2, 1, bias=True)

    def forward(self, center_ids, metadata_ids, context_ids, mask):
        """
        :param center_ids: LongTensor of batch_size
        :param metadata_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), var (batch_size, 1)
        """
        batch_size, num_context_ids = context_ids.shape
        self.lstm.flatten_parameters()

        center_embedding = self.token_embeddings(center_ids)
        metadata_embedding = self.metadata_embeddings(metadata_ids)
        context_embedding = self.token_embeddings(context_ids)

        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        metadata_embedding_tiled = metadata_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = torch.cat([center_embedding_tiled, metadata_embedding_tiled, context_embedding], dim=-1)
        merged_embeds = self.dropout(merged_embeds)
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, merged_embeds.size()[-1])
        merged_embeds.masked_fill_(mask_tiled, 0)

        h_reps, (h, _) = self.lstm(merged_embeds)
        h_sum = h.transpose(1, 0).contiguous().view(batch_size, -1)
        return self.u(h_sum), self.v(h_sum).exp()
