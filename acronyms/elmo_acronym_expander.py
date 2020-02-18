import sys

import torch
import torch.nn as nn

sys.path.insert(0, '../utils')
from compute_utils import mask_2D


class ELMoAcronymExpander(nn.Module):
    def __init__(self, elmo,):
        super(ELMoAcronymExpander, self).__init__()
        self.elmo = elmo

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
        batch_size, num_context_ids, _ = context_ids.size()
        max_output_size = lf_ids.size()[1]

        output_dim = torch.Size([batch_size, max_output_size])
        output_mask = mask_2D(output_dim, num_outputs).to('cuda')
        encoded_context = self.elmo(context_ids).mean(axis=1)
        encoded_lfs = self.elmo(lf_ids.view(batch_size * max_output_size, 5, 50)).view(
            batch_size, max_output_size, 5, -1).mean(axis=2)
        encoded_context_tiled = encoded_context.unsqueeze(1).repeat(1, max_output_size, 1)
        score = nn.CosineSimilarity(dim=-1)(encoded_context_tiled, encoded_lfs)
        score.masked_fill_(output_mask, float('-inf'))
        return score, target_lf_ids
