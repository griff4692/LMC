import sys

from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import (
    BidirectionalLanguageModelTokenEmbedder)
import numpy as np
import torch

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
from model_utils import tensor_to_np


def get_pretrained_elmo(lm_model_file='~/allennlp-0.9.0/output_path/model.tar.gz'):
    return BidirectionalLanguageModelTokenEmbedder(
        archive_file=lm_model_file,
        dropout=0.2,
        bos_eos_tokens=["<S>", "</S>"],
        remove_bos_eos=True,
        requires_grad=True
    )


def target_lf_index(target_lf, lfs):
    for i in range(len(lfs)):
        lf_tokens = lfs[i].split(';')
        for lf in lf_tokens:
            if lf.lower() == target_lf.lower():
                return i
    return -1


def process_batch(args, batcher, model, loss_func, vocab, sf_tokenized_lf_map):
    batch_input, num_outputs = batcher.next(vocab, sf_tokenized_lf_map)
    batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0), batch_input))
    scores, target, top_global_weights = model(*(batch_input + [num_outputs]),
                                               use_att=args.use_att, att_style=args.att_style)
    num_correct = len(np.where(tensor_to_np(torch.argmax(scores, 1)) == tensor_to_np(target))[0])
    num_examples = len(num_outputs)
    batch_loss = loss_func.forward(scores, target)
    return batch_loss, num_examples, num_correct, scores, top_global_weights
