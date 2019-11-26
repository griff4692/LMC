import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


''' 
##########################################INPUT##########################################

main_word : integer, the index of the main word (i.e. 333), not vector
context_word : array, the index of context words with respect to main word
n_vocab : integer, size of vocabulary
output_dim: output dim of hidden layers

##########################################OUTPUT##########################################

u: mean parameter
sigma: standard deviation
        
'''


class Encoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, args.encoder_input_dim, padding_idx=0)
        self.f = nn.Linear(args.encoder_input_dim * 2, args.encoder_hidden_dim, bias=True)
        self.u = nn.Linear(args.encoder_hidden_dim, args.latent_dim, bias=True)
        self.v = nn.Linear(args.encoder_hidden_dim, 1, bias=True)
        
    def forward(self, center_ids, context_ids):
        """
        :param center_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :return: mu (batch_size, latent_dim), logvar (batch_size, 1)
        """
        center_embedding = self.embeddings(center_ids)
        context_embedding = self.embeddings(context_ids)

        num_context_ids = context_embedding.shape[1]
        center_embedding_tiled = center_embedding.unsqueeze(1).repeat(1, num_context_ids, 1)
        merged_embeds = torch.cat([center_embedding_tiled, context_embedding], dim=-1)
        
        h = F.relu(self.f(merged_embeds)).sum(1)
        return self.u(h), self.v(h).exp()
