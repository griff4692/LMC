import torch.nn as nn

from model.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, args, vocab_size ):
        super(VAE, self).__init__()

        self.encoder = Encoder(args, vocab_size)

    def forward(self, center_ids, context_ids):
        return self.encoder(center_ids, context_ids)
