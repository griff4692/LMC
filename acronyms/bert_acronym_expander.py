import sys

import torch
import torch.nn as nn

sys.path.insert(0, '../utils')
from compute_utils import mask_2D


class BertAcronymExpander(nn.Module):
    def __init__(self, bert_pre_trained_model, cls_only=True):
        super(BertAcronymExpander, self).__init__()
        self.bert_pre_trained_model = bert_pre_trained_model
        self.cls_only = cls_only

    def forward(self, context_ids, lf_ids, target_lf_ids, num_outputs):
        """
        :param sf_ids: batch_size
        :param context_ids: batch_size, num_context_ids, 50
        :param lf_ids: batch_size, max_output_size, max_lf_len, 50
        :param lf_token_ct: batch_size, max_output_size - normalizer for lf_ids
        :param target_lf_ids: batch_size
        :param num_outputs: batch_size
        :return:
        """
        batch_size, num_context_ids = context_ids.size()
        max_output_size = lf_ids.size()[1]
        # output_mask = mask_2D(output_dim, num_outputs).to('cuda')

        hidden_states, _ = self.bert_pre_trained_model(context_ids)[-2:]
        hidden_states = torch.stack(hidden_states)
        hidden_states = hidden_states.permute(1,2,0,3)
        
        # CLS token top layer
        if self.cls_only:
            context_vector = hidden_states[:,0,-1,:]
        else:
            context_vector = hidden_states[:,:,-1,:].mean(axis=1)

        lfs_hidden_states, _ = self.bert_pre_trained_model(lf_ids.view(batch_size * max_output_size, -1))[-2:]
        lfs_hidden_states = torch.stack(lfs_hidden_states)
        lfs_hidden_states = lfs_hidden_states.permute(1,2,0,3)
        lfs_hidden_states = lfs_hidden_states.view(batch_size, max_output_size, -1, 13, 768)

        if self.cls_only:
            lfs_vector = lfs_hidden_states[:,:,0,-1,:]
        else:
            lfs_vector = lfs_hidden_states[:,:,:,-1,:].mean(axis=2)

        context_vector_tiled = context_vector.unsqueeze(1).repeat(1, max_output_size, 1)
        score = nn.CosineSimilarity(dim=-1)(context_vector_tiled, lfs_vector)
        return score, target_lf_ids
