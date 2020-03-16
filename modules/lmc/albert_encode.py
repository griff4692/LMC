import os
import sys

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from compute_utils import compute_att


def encode(model, **kwargs):
    if model.pool_layers:
        all_encoded_layers = model.bert(**kwargs)[2]
        num_layers = len(all_encoded_layers)
        last_layers = min(num_layers - 1, 4)
        output = torch.stack(all_encoded_layers[-last_layers:]).sum(0)
    else:
        output = model.bert(**kwargs)[0]
    att_mask = kwargs['attention_mask'] == 0
    return compute_att(output, att_mask, model.att_linear)
