import sys

import numpy as np
import torch

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
from model_utils import tensor_to_np


def target_lf_index(target_lf, lfs):
    for i in range(len(lfs)):
        lf_tokens = lfs[i].split(';')
        for lf in lf_tokens:
            if lf.lower() == target_lf.lower():
                return i
    return -1


def process_batch(batcher, model, loss_func, vocab, sf_tokenized_lf_map):
    batch_input, num_outputs = batcher.next(vocab, sf_tokenized_lf_map)
    batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0), batch_input))
    proba, target, var = model(*(batch_input + [num_outputs]))
    num_correct = len(np.where(tensor_to_np(torch.argmax(proba, 1)) == tensor_to_np(target))[0])
    num_examples = len(num_outputs)
    batch_loss = loss_func.forward(proba, target)
    return batch_loss, num_examples, num_correct, proba, var