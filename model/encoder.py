import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import numpy as np

class Encoder(nn.Module):
    def __init__(self, args, vocab_size,doc_size):
        super(Encoder, self).__init__()
        self.vocab_embeddings = nn.Embedding(vocab_size, args.encoder_input_dim, padding_idx=0)
        vocab_embeddings_init = np.random.uniform(low=0, high=3, size=(vocab_size, args.encoder_input_dim))
        self.vocab_embeddings.load_state_dict({'weight': torch.from_numpy(vocab_embeddings_init)})
        
        
        self.doc_embeddings = nn.Embedding(doc_size+1, args.encoder_input_dim, padding_idx=0)
        doc_embeddings_init = np.random.uniform(low=0, high=3, size=(vocab_size, args.encoder_input_dim))
        self.doc_embeddings.load_state_dict({'weight': torch.from_numpy(doc_embeddings_init)})
        
        self.f = nn.Linear(args.encoder_input_dim * 2, args.encoder_hidden_dim, bias=True)
        self.u = nn.Linear(args.encoder_hidden_dim, args.latent_dim, bias=True)
        self.v = nn.Linear(args.encoder_hidden_dim, 1, bias=True)
        
    def forward(self, center_ids, selected_doc_ids):
        """
        :param center_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size x 2 * context_window
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), logvar (batch_size, 1)
        """
        center_embedding = self.vocab_embeddings(center_ids)
        document_embedding = self.doc_embeddings(selected_doc_ids)
        
        merged_embeds = torch.cat([center_embedding, document_embedding], dim=-1)
        h = F.relu(self.f(merged_embeds))
        
        scalar = torch.FloatTensor([1]).cuda()
        scalar.expand_as(center_ids)
        
        return self.u(h), torch.max(scalar,self.v(h).exp())
