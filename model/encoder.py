import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data


class Encoder(nn.Module):
    def __init__(self, args, vocab_size, var_dim):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, args.encoder_input_dim, padding_idx=0)
        self.f = nn.Linear(args.encoder_input_dim * 2, args.encoder_hidden_dim, bias=True)
        self.u = nn.Linear(args.encoder_hidden_dim, args.latent_dim, bias=True)
        self.v = nn.Linear(args.encoder_hidden_dim, var_dim, bias=True)
        
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

        h_reps = F.relu(self.f(merged_embeds))
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, h_reps.size()[-1])
        h_reps.masked_fill_(mask_tiled, 0)
        h = h_reps.sum(1)
        return self.u(h), self.v(h).exp()
