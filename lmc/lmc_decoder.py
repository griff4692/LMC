import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data


class LMCDecoder(nn.Module):
    def __init__(self, token_vocab_size, section_vocab_size, input_dim=100, hidden_dim=64, output_dim=100):
        super(LMCDecoder, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.f = nn.Linear(input_dim * 2, hidden_dim, bias=True)
        self.u = nn.Linear(hidden_dim, output_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=True)

        self.token_embeddings = nn.Embedding(token_vocab_size, input_dim, padding_idx=0)
        self.metadata_embeddings = nn.Embedding(section_vocab_size, input_dim, padding_idx=0)

    def forward(self, center_ids, metadata_ids, normalizer=None):
        """
        :param center_ids: LongTensor of batch_size
        :param metadata_ids: LongTensor of batch_size
        :return: mu (batch_size, latent_dim), var (batch_size, 1)
        """
        center_embedding = self.token_embeddings(center_ids)
        if len(center_ids.size()) > len(metadata_ids.size()):
            center_embedding = center_embedding.sum(1) / normalizer

        metadata_embedding = self.metadata_embeddings(metadata_ids)
        merged_embeds = self.dropout(torch.cat([center_embedding, metadata_embedding], dim=-1))
        h = self.dropout(F.relu(self.f(merged_embeds)))
        return self.u(h), self.v(h).exp()
