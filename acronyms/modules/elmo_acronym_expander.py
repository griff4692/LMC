import os
import sys

import numpy as np
import torch
import torch.nn as nn

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from compute_utils import mask_2D


class ELMoAcronymExpander(nn.Module):
    def __init__(self, elmo,):
        super(ELMoAcronymExpander, self).__init__()
        self.elmo = elmo

    def forward(self, context_ids, context_token_ct, lf_ids, lf_token_ct, target_lf_ids, num_outputs):
        """
        :param sf_ids: batch_size
        :param context_ids: batch_size, num_context_ids, 50
        :param lf_ids: batch_size, max_output_size, max_lf_len, 50
        :param lf_token_ct: batch_size, max_output_size - normalizer for lf_ids
        :param target_lf_ids: batch_size
        :param num_outputs: batch_size
        :return: LF predictions, LF ground truths
        """
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size, num_context_ids, _ = context_ids.size()
        _, max_output_size, max_lf_len, _ = lf_ids.size()

        context_mask = np.ones([batch_size, num_context_ids], dtype=int)
        lf_mask = np.ones([batch_size, max_output_size, max_lf_len], dtype=int)
        for i in range(batch_size):
            context_mask[i, context_token_ct[i]:] = 0
            for lf_idx in range(max_output_size):
                lf_mask[i, lf_idx, lf_token_ct[i, lf_idx]:] = 0
        context_mask = torch.FloatTensor(context_mask).to(device_str)
        lf_mask = torch.FloatTensor(lf_mask).to(device_str)

        output_dim = torch.Size([batch_size, max_output_size])
        output_mask = mask_2D(output_dim, num_outputs).to(device_str)
        encoded_context = self.elmo(context_ids)
        encoded_context = (encoded_context * context_mask.unsqueeze(-1)).sum(axis=1) / context_token_ct.unsqueeze(-1)
        encoded_lfs = self.elmo(lf_ids.view(batch_size * max_output_size, max_lf_len, 50)).view(batch_size, max_output_size, max_lf_len, -1)

        encoded_lfs = (encoded_lfs * lf_mask.unsqueeze(-1)).sum(2) / lf_token_ct.unsqueeze(-1)
        encoded_context_tiled = encoded_context.unsqueeze(1).repeat(1, max_output_size, 1)
        score = nn.CosineSimilarity(dim=-1)(encoded_context_tiled, encoded_lfs)
        score.masked_fill_(output_mask, float('-inf'))
        return score, target_lf_ids
