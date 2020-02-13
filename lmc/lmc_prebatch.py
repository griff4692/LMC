import pickle
import os
import sys

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_sections import enumerate_metadata_ids_lmc
from model_utils import get_git_revision_hash, render_args
from vocab import Vocab


def _get_metadata_id_sample(token_metadata_samples, token_id):
    counter, sids = token_metadata_samples[token_id]
    sid = sids[counter]
    token_metadata_samples[token_id][0] += 1
    if token_metadata_samples[token_id][0] >= len(sids):
        np.random.shuffle(sids)
        token_metadata_samples[token_id] = [0, sids]
    return sid


def extract_context_ids(ids, center_idx, target_window):
    start_idx = max(0, center_idx - target_window)
    end_idx = min(len(ids), center_idx + target_window + 1)

    left_context = ids[start_idx:center_idx]
    right_context = ids[center_idx + 1:end_idx]

    metadata_boundary_left = np.where(left_context == -1)[0]
    metadata_boundary_right = np.where(right_context == -1)[0]

    left_trunc_idx = 0 if len(metadata_boundary_left) == 0 else metadata_boundary_left[-1] + 1
    right_trunc_idx = len(right_context) if len(metadata_boundary_right) == 0 else metadata_boundary_right[0]

    left_context_truncated = left_context[left_trunc_idx:]
    right_context_truncated = right_context[:right_trunc_idx]

    return np.concatenate([left_context_truncated, right_context_truncated])


def data_to_tens(ids, all_metadata_pos_idxs, full_metadata_ids, token_metadata_samples, token_vocab, window_size):
    N = len(ids)
    batch_idxs = np.array(list(set(np.arange(N)) - set(all_metadata_pos_idxs)))
    N = len(batch_idxs)
    center_ids = ids[batch_idxs]
    context_ids = np.zeros([N, (window_size * 2)], dtype=int)
    neg_ids = token_vocab.neg_sample(size=(N, (window_size * 2)))

    num_metadata = token_metadata_samples[1][1].shape[1]
    center_metadata_ids = np.zeros([N, ], dtype=int)
    context_metadata_ids = np.zeros([N, (window_size * 2), num_metadata], dtype=int)
    neg_metadata_ids = np.zeros([N, (window_size * 2), num_metadata], dtype=int)

    window_sizes = []
    for batch_idx, center_idx in enumerate(batch_idxs):
        example_context_ids = extract_context_ids(ids, center_idx, window_size)
        center_metadata_ids[batch_idx] = full_metadata_ids[center_idx]
        context_ids[batch_idx, :len(example_context_ids)] = example_context_ids
        window_sizes.append(len(example_context_ids))
        for idx, context_id in enumerate(example_context_ids):
            n_id = neg_ids[batch_idx, idx]
            c_sids = _get_metadata_id_sample(token_metadata_samples, context_id)
            n_sids = _get_metadata_id_sample(token_metadata_samples, n_id)
            context_metadata_ids[batch_idx, idx, :] = c_sids
            neg_metadata_ids[batch_idx, idx, :] = n_sids

    return (center_ids, center_metadata_ids, context_ids, context_metadata_ids, neg_ids, neg_metadata_ids,
            window_sizes)


def generate_metadata_samples(token_metadata_counts, metadata_vocab, token_vocab, sample=5):
    token_metadata_samples = {}
    smooth_counts = np.zeros([metadata_vocab.size()])
    all_metadata_ids = np.arange(metadata_vocab.size())
    for k, (sids, sp) in token_metadata_counts.items():
        size = [min(len(sp) * 100, 10000), sample]
        # Add smoothing
        smooth_counts.fill(1.0)
        smooth_counts[sids] += sp
        smooth_p = smooth_counts / smooth_counts.sum()
        rand_sids = np.random.choice(all_metadata_ids, size=size, replace=True, p=smooth_p)
        start_idx = 0
        token_metadata_samples[k] = [start_idx, rand_sids]

    token_vocab.truncate(token_vocab.section_start_vocab_id)
    return token_metadata_samples


def precompute(args):
    # Load Data
    debug_str = '_mini' if args.debug else ''

    ids_infile = os.path.join(args.data_dir, 'ids{}.npy'.format(debug_str))
    print('Loading data from {}...'.format(ids_infile))
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)

    # Load Vocabulary
    vocab_infile = os.path.join(args.data_dir, 'vocab{}.pk'.format(debug_str))
    print('Loading vocabulary from {}...'.format(vocab_infile))
    with open(vocab_infile, 'rb') as fd:
        token_vocab = pickle.load(fd)
    print('Loaded vocabulary of size={}...'.format(token_vocab.section_start_vocab_id))

    print('Collecting metadata information for {}...'.format(args.metadata))
    assert token_vocab.section_start_vocab_id <= token_vocab.category_start_vocab_id
    start_id = token_vocab.section_start_vocab_id if args.metadata == 'section' else token_vocab.category_start_vocab_id
    end_id = token_vocab.category_start_vocab_id if args.metadata == 'section' else token_vocab.size()
    metadata_id_range = np.arange(start_id, end_id)
    is_metadata = np.isin(ids, metadata_id_range)
    metadata_pos_idxs = np.where(is_metadata)[0]
    all_metadata_pos_idxs = np.where(ids >= token_vocab.section_start_vocab_id)[0]
    metadata_vocab = Vocab()
    for id in metadata_id_range:
        name = token_vocab.get_token(id)
        metadata_vocab.add_token(name)
    full_metadata_ids, token_metadata_counts = enumerate_metadata_ids_lmc(
        ids, metadata_pos_idxs, token_vocab, metadata_vocab)
    ids[all_metadata_pos_idxs] = -1

    token_metadata_samples = generate_metadata_samples(
        token_metadata_counts, metadata_vocab, token_vocab, sample=args.metadata_samples)
    data = data_to_tens(ids, all_metadata_pos_idxs, full_metadata_ids, token_metadata_samples, token_vocab, args.window)
    data_tens = list(map(torch.LongTensor, data))
    data_loader = DataLoader(TensorDataset(*data_tens), shuffle=True, batch_size=args.batch_size)

    out_fn = '../preprocess/data/batches{}.pth'.format(debug_str)
    print('Saving batches to {}'.format(out_fn))
    output = {
        'data_loader': data_loader,
        'token_vocab': token_vocab,
        'metadata_vocab': metadata_vocab,
        'token_metadata_counts': token_metadata_counts
    }
    torch.save(output,  out_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Precompute Batches for LMC')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--window', default=10, type=int)

    # Model Hyperparameters
    parser.add_argument('--metadata_samples', default=5, type=int)
    parser.add_argument('--metadata', default='section',
                        help='sections or category. What to define latent variable over.')
    args = parser.parse_args()
    args.git_hash = get_git_revision_hash()
    render_args(args)
    precompute(args)
