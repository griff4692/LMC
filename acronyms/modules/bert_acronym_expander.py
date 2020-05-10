import os
import sys

import numpy as np
import torch
import torch.nn as nn

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from compute_utils import mask_2D


class BERTAcronymExpander(nn.Module):
    def __init__(self, bert_model):
        super(BERTAcronymExpander, self).__init__()
        self.bert = bert_model

    def encode(self, ids, mask, token_cts):
        pool = False
        encodings = self.bert(input_ids=ids, attention_mask=mask)
        if pool:
            all_encoded_layers = encodings[2]
            num_layers = len(all_encoded_layers)
            last_layers = min(num_layers - 1, 4)
            output = torch.stack(all_encoded_layers[-last_layers:]).sum(0)
        else:
            output = encodings[0]
        mask_full = mask.unsqueeze(-1).repeat(1, 1, 768)
        output *= mask_full
        output_mean = output.sum(1) / token_cts.unsqueeze(1)
        return output_mean

    def forward(self, context_ids, context_token_ct, lf_ids, lf_token_ct, target_lf_ids, num_outputs):
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size, num_context_ids = context_ids.size()
        _, max_output_size, max_lf_len = lf_ids.size()

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
        encoded_context = self.encode(context_ids, context_mask, context_token_ct)

        lf_ids_flat = lf_ids.view(batch_size * max_output_size, max_lf_len)
        lf_mask_flat = lf_mask.view(batch_size * max_output_size, max_lf_len)
        lf_token_ct_flat = lf_token_ct.view(batch_size * max_output_size)
        encoded_lfs_flat = self.encode(lf_ids_flat, lf_mask_flat, lf_token_ct_flat)
        encoded_lfs = encoded_lfs_flat.view(batch_size, max_output_size, -1)

        encoded_context_tiled = encoded_context.unsqueeze(1).repeat(1, max_output_size, 1)
        score = nn.CosineSimilarity(dim=-1)(encoded_context_tiled, encoded_lfs)
        score.masked_fill_(output_mask, float('-inf'))
        return score, target_lf_ids
