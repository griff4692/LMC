import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from compute_utils import kl_spher, mask_2D
from torch.distributions.multivariate_normal import MultivariateNormal

from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, args, vocab_size,doc_size, pretrained_embeddings=None):
        super(VAE, self).__init__()

        self.encoder = Encoder(args, vocab_size,doc_size)
        self.margin = args.hinge_loss_margin or 1.0

    def forward(self, center_ids, context_ids, neg_context_ids, num_contexts, device, selected_doc_ids):
        """
        :param center_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :param neg_context_ids: batch_size, 2 * context_window
        :param num_contexts: batch_size (how many context words for each center id - necessary for masking padding)
        :return: cost components: KL-Divergence (q(z|w,c) || p(z|w)) and max margin (reconstruction error)
        """
        
        # Mask padded context ids
        loss = 0
        batch_size = center_ids.size()[0]
        mu_center, sigma_center = self.encoder(center_ids, selected_doc_ids)
        
        for i in range(context_ids.shape[1]):
            
            mu_pos_context, sigma_pos_context =self.encoder(context_ids[:,i], selected_doc_ids[i].repeat(batch_size))
            mu_neg_context, sigma_neg_context =self.encoder(neg_context_ids[:,i], selected_doc_ids[i].repeat(batch_size))
            
            kl_pos =  kl_spher(mu_center, sigma_center, mu_pos_context, sigma_pos_context)
            kl_neg =  kl_spher(mu_center, sigma_center, mu_neg_context, sigma_neg_context)
            kl = kl_pos - kl_neg
                        
            hinge_loss = (kl + self.margin).clamp_min_(0)
            loss += hinge_loss
            
        return loss.sum()
