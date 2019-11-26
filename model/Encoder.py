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
    
    def __init__(self,vocab_size,embedding_size,hidden_dim,latent_dim):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_size,padding_idx = 0)
        self.f = nn.Linear(vocab_size,hidden_dim,bias = True)
        self.u = nn.Linear(hidden_dim,latent_dim,bias = True)
        self.v = nn.Linear(hidden_dim,1,bias = True)
                
        #nn.embedings#
        # define R as a tensor batch size,(window_size+1)*vocab_size
        
        
    def forward(self,context_ids, center_ids):
        #batch size* window size
        #batch size*1
        #context embedings = self.contextembedings * self.
        
        center_embedding = self.embeddings(center_ids)
        context_embedding = self.embeddings(context_ids)
        
        R = torch.cat(center_embedding.repeat(1,context_embedding.shape[1]),context_embedding, dim = 1)
        
        h =  F.relu(self.f(R))
        mu = self.u(h)
        logvar = self.v(h)
        
        return mu, logvar.exp()
    
    
    
    
    
    
    