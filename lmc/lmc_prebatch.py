import itertools
import os
import pickle
from shutil import rmtree
import sys

import argparse
import numpy as np
import ray
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_sections import enumerate_metadata_ids_lmc
from model_utils import get_git_revision_hash, render_args
from vocab import Vocab


def _get_token(id, i2w):
    return i2w[id]


def _get_tokens(ids, i2w):
    return list(map(lambda id: _get_token(id, i2w), ids))


class DistributedDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_files = os.listdir(self.base_dir)
        np.sort(self.data_files)

    def __getitem__(self, idx):
        fn = os.path.join(self.base_dir, self.data_files[idx])
        return torch.load(fn)

    def __len__(self):
        return len(self.data_files)


def create_tokenizer_maps(bert_tokenizer, token_vocab, metadata_vocab):
    token_to_wp = [None] * token_vocab.size()
    meta_to_wp = [None] * metadata_vocab.size()

    for i in range(token_vocab.size()):
        token = token_vocab.get_token(i)
        wps = bert_tokenizer.encode(token, add_special_tokens=False)
        token_to_wp[i] = wps
    for i in range(metadata_vocab.size()):
        meta = metadata_vocab.get_token(i)
        wps = bert_tokenizer.encode(meta, add_special_tokens=False)
        meta_to_wp[i] = wps

    special_map = {}
    for t, i in zip(bert_tokenizer.all_special_tokens, bert_tokenizer.all_special_ids):
        if 'header' not in t and 'document' not in t:
            special_map[t] = i
    return {'token_to_wp': token_to_wp, 'meta_to_wp': meta_to_wp, 'special_to_wp': special_map}


def _get_metadata_id_sample(token_metadata_samples, token_id):
    counter, sids = token_metadata_samples[token_id]
    sid = sids[counter]
    token_metadata_samples[token_id][0] += 1
    if token_metadata_samples[token_id][0] >= len(sids):
        token_metadata_samples[token_id] = [0, sids]
    return sid


def extract_full_context_ids(ids, center_idx, target_window):
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

    return list(left_context_truncated), list(right_context_truncated)


@ray.remote
def _process_bert_batch_remote(all_batch_cts, **kwargs):
    _process_bert_batch(all_batch_cts, **kwargs)


def _process_bert_batch(all_batch_cts, **kwargs):
    batches = kwargs['batches']
    window_size = kwargs['window_size']
    debug_str = kwargs['debug_str']
    ids = kwargs['ids']
    neg_sample_p = kwargs['neg_sample_p']
    full_metadata_ids = kwargs['full_metadata_ids']
    token_metadata_samples = kwargs['token_metadata_samples']
    chunk_out_fn = '../preprocess/data/pre/chunks_bert{}'.format(debug_str)
    num_batches = len(all_batch_cts)
    wp_conversions = kwargs['wp_conversions']
    token_to_wp = wp_conversions['token_to_wp']
    meta_to_wp = wp_conversions['meta_to_wp']
    special_to_wp = wp_conversions['special_to_wp']

    PAD_ID = special_to_wp['[PAD]']
    CLS_ID = special_to_wp['[CLS]']
    SEP_ID = special_to_wp['[SEP]']

    num_metadata = token_metadata_samples[1][1].shape[-1]
    max_single_len = num_metadata + 2 + 5  # 2 for special tokens, 5 for max # of wps for unigram
    max_encoder_len = int((window_size * 2 + 1) * 2.5) + 2  # max avg of 2.5 wps ids per unigram in context window

    for batch_ct in tqdm(all_batch_cts, total=num_batches):
        batch_idxs = batches[batch_ct, :]
        batch_size = len(batch_idxs)
        center_ids = ids[batch_idxs]

        neg_ids = np.random.choice(np.arange(len(neg_sample_p)), size=(batch_size, 2 * window_size), p=neg_sample_p)

        center_bert_ids = np.zeros([batch_size, max_single_len], dtype=int)
        center_bert_ids.fill(PAD_ID)
        center_bert_mask = np.ones([batch_size, max_single_len])

        pos_bert_ids = np.zeros([batch_size, window_size * 2, max_single_len], dtype=int)
        neg_bert_ids = np.zeros([batch_size, window_size * 2, max_single_len], dtype=int)
        pos_bert_mask = np.ones([batch_size, window_size * 2, max_single_len])
        neg_bert_mask = np.ones([batch_size, window_size * 2, max_single_len])
        pos_bert_ids.fill(PAD_ID)
        neg_bert_ids.fill(PAD_ID)

        window_sizes = []

        context_bert_ids = np.zeros([batch_size, max_encoder_len], dtype=int)
        context_bert_ids.fill(PAD_ID)
        context_bert_mask = np.ones([batch_size, max_encoder_len], dtype=int)
        context_token_type_ids = np.zeros([batch_size, max_encoder_len], dtype=int)

        for batch_idx, center_idx in enumerate(batch_idxs):
            center_id = center_ids[batch_idx]
            left_context_ids, right_context_ids = extract_full_context_ids(ids, center_idx, window_size)
            L, R = len(left_context_ids), len(right_context_ids)

            center_tok_wp_ids = token_to_wp[center_id]
            center_meta_wp_ids = meta_to_wp[full_metadata_ids[center_idx]]

            left_wp_ids = list(map(lambda id: token_to_wp[id], left_context_ids))
            right_wp_ids = list(map(lambda id: token_to_wp[id], right_context_ids))

            window_sizes.append(L + R)

            center_seq_bert = [CLS_ID] + center_meta_wp_ids + [SEP_ID] + center_tok_wp_ids
            center_encoded_len = min(len(center_seq_bert), max_single_len)
            center_bert_ids[batch_idx, :center_encoded_len] = center_seq_bert[:center_encoded_len]
            center_bert_mask[batch_idx, center_encoded_len:] = 0

            full_ids = left_context_ids + right_context_ids
            full_wp_ids = left_wp_ids + right_wp_ids

            flattened_left = list(itertools.chain(*left_wp_ids))
            flattened_right = list(itertools.chain(*right_wp_ids))
            c_wp_seq = flattened_left + center_tok_wp_ids + flattened_right
            left_seq = center_meta_wp_ids + center_tok_wp_ids
            full_context_wp_seq = ([CLS_ID] + c_wp_seq + [SEP_ID] + left_seq + [SEP_ID])

            cutoff = min(len(full_context_wp_seq), max_encoder_len)
            context_bert_ids[batch_idx, :cutoff] = full_context_wp_seq[:cutoff]
            context_bert_mask[batch_idx, cutoff:] = 1
            left_cutoff = 1 + len(c_wp_seq)
            context_token_type_ids[batch_idx, left_cutoff:] = 1

            for idx, (context_id, p_wp_ids) in enumerate(zip(full_ids, full_wp_ids)):
                n_id = neg_ids[batch_idx, idx]
                n_wp_ids = token_to_wp[n_id]

                p_sids = _get_metadata_id_sample(token_metadata_samples, context_id)
                n_sids = _get_metadata_id_sample(token_metadata_samples, n_id)

                p_wp_sids = list(map(lambda x: meta_to_wp[x][0], p_sids))
                n_wp_sids = list(map(lambda x: meta_to_wp[x][0], n_sids))

                pos_seq_wp = [CLS_ID] + p_wp_ids + [SEP_ID] + p_wp_sids + [SEP_ID]
                neg_seq_wp = [CLS_ID] + n_wp_ids + [SEP_ID] + n_wp_sids + [SEP_ID]

                pos_encoded_len = min(len(pos_seq_wp), max_single_len)
                neg_encoded_len = min(len(neg_seq_wp), max_single_len)

                pos_bert_ids[batch_idx, idx, :pos_encoded_len] = pos_seq_wp[:pos_encoded_len]
                neg_bert_ids[batch_idx, idx, :neg_encoded_len] = neg_seq_wp[:neg_encoded_len]
                pos_bert_mask[batch_idx, idx, pos_encoded_len:] = 0
                neg_bert_mask[batch_idx, idx, neg_encoded_len:] = 0

        batch_ids = [context_bert_ids, center_bert_ids, pos_bert_ids, neg_bert_ids, context_token_type_ids,
                           window_sizes]
        batch_ids = list(map(torch.LongTensor, batch_ids))
        batch_masks = [context_bert_mask, center_bert_mask, pos_bert_mask, neg_bert_mask]
        batch_masks = list(map(torch.FloatTensor, batch_masks))
        batch_data = batch_ids + batch_masks
        torch.save(batch_data, '{}/{}.pth'.format(chunk_out_fn, batch_ct))
    return chunk_out_fn


def data_to_tens_bert(args, ids, all_metadata_pos_idxs, full_metadata_ids, token_metadata_samples, token_vocab, window_size,
                      metadata_vocab=None, tokenizer=None):
    debug_str = '_mini' if args.debug else ''
    metadata_tokens = metadata_vocab.i2w[1:] + ['digitparsed']
    special_tokens_dict = {'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]', 'bos_token': '[BOS]',
                           'eos_token': '[EOS]', 'pad_token': '[PAD]', 'mask_token': '[MASK]',
                           'additional_special_tokens': metadata_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('Readded special tokens={}'.format(num_added_toks))
    print('Mapping regular vocab ids to WordPiece ids for token and metadata...')
    wp_conversions = create_tokenizer_maps(tokenizer, token_vocab, metadata_vocab)
    num_ids = len(ids)
    print('Shuffling data...')
    all_batch_idxs = np.array(list(set(np.arange(num_ids)) - set(all_metadata_pos_idxs)), dtype=int)
    np.random.shuffle(all_batch_idxs)
    num_batches = len(all_batch_idxs) // args.batch_size
    truncated_N = args.batch_size * num_batches
    batches = all_batch_idxs[:truncated_N].reshape(num_batches, args.batch_size)

    token_vocab.neg_sample(size=(1,))
    neg_sample_p = token_vocab.cached_neg_sample_prob

    kwargs = {
        'batches': batches,
        'full_metadata_ids': full_metadata_ids,
        'token_metadata_samples': token_metadata_samples,
        'neg_sample_p': neg_sample_p,
        'window_size': window_size,
        'full_metadata_ids': full_metadata_ids,
        'ids': ids,
        'debug_str': debug_str,
        'wp_conversions': wp_conversions
    }

    shared_mem_limit = 1e+9 if args.debug else 1e+11
    if not args.sequential:
        ray.init(num_cpus=args.num_cpus, memory=shared_mem_limit, object_store_memory=shared_mem_limit)
        for k, v in kwargs.items():
            kwargs[k] = ray.put(v, weakref=True)

        print('Using {} CPUS'.format(args.num_cpus))
        step_size = max(num_batches // args.num_cpus, 1)
        for step_idx in tqdm(range(0, num_batches, step_size)):
            results = []
            batch_idxs = list(range(step_idx, min(step_idx + step_size, num_batches)))
            result = _process_bert_batch_remote.remote(batch_idxs, **kwargs)
            results.append(result)
        ray.get(results)
    else:
        x = list(map(lambda i: _process_bert_batch([i], **kwargs), tqdm(list(range(num_batches)))))
        print(len(x))


def _process_batch(all_batch_cts, **kwargs):
    window_size = kwargs['window_size']
    debug_str = kwargs['debug_str']
    ids = kwargs['ids']
    neg_sample_p = kwargs['neg_sample_p']
    full_metadata_ids = kwargs['full_metadata_ids']
    token_metadata_samples = kwargs['token_metadata_samples']
    chunk_out_fn = '../preprocess/data/pre/chunks{}'.format(debug_str)
    num_batches = len(all_batch_cts)
    for batch_ct in tqdm(all_batch_cts, total=num_batches):
        num_metadata = kwargs['token_metadata_samples'][1][1].shape[1]
        batch_idxs = kwargs['batches'][batch_ct, :]
        batch_size = len(batch_idxs)

        neg_ids = np.random.choice(np.arange(len(neg_sample_p)), size=(batch_size, 2 * window_size), p=neg_sample_p)
        center_ids = ids[batch_idxs]
        context_ids = np.zeros([batch_size, (window_size * 2)], dtype=int)

        center_metadata_ids = np.zeros([batch_size, ], dtype=int)
        context_metadata_ids = np.zeros([batch_size, (window_size * 2), num_metadata], dtype=int)
        neg_metadata_ids = np.zeros([batch_size, (window_size * 2), num_metadata], dtype=int)
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

        data = (center_ids, center_metadata_ids, context_ids, context_metadata_ids, neg_ids, neg_metadata_ids,
                window_sizes)
        data_tens = list(map(torch.LongTensor, data))
        if batch_ct % 10000 == 0:
            print('Processing batch={}'.format(batch_ct))
        torch.save(data_tens, '{}/{}.pth'.format(chunk_out_fn, batch_ct))
    return chunk_out_fn


@ray.remote
def _process_batch_remote(all_batch_cts, **kwargs):
    return _process_batch(all_batch_cts, **kwargs)


def data_to_tens(args, ids, all_metadata_pos_idxs, full_metadata_ids, token_metadata_samples, token_vocab, window_size,
                 metadata_vocab=None, tokenizer=None):
    debug_str = '_mini' if args.debug else ''
    num_ids = len(ids)
    print('Shuffling data idxs...')
    all_batch_idxs = np.array(list(set(np.arange(num_ids)) - set(all_metadata_pos_idxs)))
    np.random.shuffle(all_batch_idxs)
    num_batches = len(all_batch_idxs) // args.batch_size
    truncated_N = args.batch_size * num_batches
    batches = all_batch_idxs[:truncated_N].reshape(num_batches, args.batch_size)

    token_vocab.neg_sample(size=(1,))
    neg_sample_p = token_vocab.cached_neg_sample_prob

    shared_mem_limit = 1e+9 if args.debug else 1e+11

    if not args.sequential:
        ray.init(num_cpus=args.num_cpus, memory=shared_mem_limit, object_store_memory=shared_mem_limit)
        kwargs = {
            'batches': ray.put(batches),
            'full_metadata_ids': ray.put(full_metadata_ids, weakref=True),
            'token_metadata_samples': ray.put(token_metadata_samples, weakref=True),
            'neg_sample_p': ray.put(neg_sample_p, weakref=True),
            'window_size': ray.put(window_size, weakref=True),
            'full_metadata_ids': ray.put(full_metadata_ids, weakref=True),
            'ids': ray.put(ids, weakref=True),
            'debug_str': ray.put(debug_str, weakref=True)
        }

        print('Using {} CPUS'.format(CPU_COUNT))
        step_size = max(num_batches // args.num_cpus, 1)
        for step_idx in tqdm(range(0, num_batches, step_size)):
            results = []
            batch_idxs = list(range(step_idx, min(step_idx + step_size, num_batches)))
            result = _process_batch_remote.remote(batch_idxs, **kwargs)
            results.append(result)
        ray.get(results)
    else:
        x = list(map(lambda i: _process_batch([i]), tqdm(list(range(num_batches)))))
        print(len(x))


def extract_context_ids(ids, center_idx, target_window):
    l, r = extract_full_context_ids(ids, center_idx, target_window)
    return np.concatenate([l, r])


def generate_metadata_samples(token_metadata_counts, metadata_vocab, sample=5):
    token_metadata_samples = {}
    smooth_counts = np.zeros([metadata_vocab.size()])
    all_metadata_ids = np.arange(metadata_vocab.size())
    for k, (sids, sp) in tqdm(token_metadata_counts.items(), total=len(token_metadata_counts)):
        size = [min(len(sp) * 10, 1000), sample]
        # Add smoothing
        smooth_counts.fill(1.0)
        smooth_counts[sids] += sp
        smooth_p = smooth_counts / smooth_counts.sum()
        rand_sids = np.random.choice(all_metadata_ids, size=size, replace=True, p=smooth_p)
        start_idx = 0
        token_metadata_samples[k] = [start_idx, rand_sids]
    return token_metadata_samples


def precompute(args, loader):
    # Load Data
    debug_str = '_mini' if args.debug else ''
    bert_str = '_bert' if args.bert else ''
    out_info_fn = '../preprocess/data/pre/batch_info{}{}.pth'.format(bert_str, debug_str)
    chunk_dir = '../preprocess/data/pre/chunks{}{}'.format(bert_str, debug_str)

    if os.path.exists(out_info_fn):
        print('Removing previous metadata from {}'.format(out_info_fn))
        os.remove(out_info_fn)

    if os.path.exists(chunk_dir):
        print('Removing previous directory holding preprocessed batches={}'.format(chunk_dir))
        rmtree(chunk_dir)
    print('Recreating the directory...')
    os.mkdir(chunk_dir)

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
    all_metadata_pos_idxs = np.where(ids >= token_vocab.section_start_vocab_id)[0]
    metadata_vocab = Vocab()
    for id in metadata_id_range:
        name = token_vocab.get_token(id)
        metadata_vocab.add_token(name)
    print('Enumerating over ids to get per id metadata info...')
    metadata_pos_idxs = np.where(is_metadata)[0]
    full_metadata_ids, token_metadata_counts = enumerate_metadata_ids_lmc(
        ids, metadata_pos_idxs, token_vocab, metadata_vocab)
    ids[all_metadata_pos_idxs] = -1

    print('Generating samples...')
    token_metadata_samples = generate_metadata_samples(
        token_metadata_counts, metadata_vocab, sample=args.metadata_samples)
    token_vocab.truncate(token_vocab.section_start_vocab_id)
    tokenizer_fn = '../preprocess/data/tokenizer_vocab.pth'
    tokenizer = None
    if os.path.exists(tokenizer_fn):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_fn)
    loader(args, ids, all_metadata_pos_idxs, full_metadata_ids, token_metadata_samples, token_vocab, args.window,
           metadata_vocab=metadata_vocab, tokenizer=tokenizer)

    print('Saving batches to {}'.format(out_info_fn))
    output = {
        'token_vocab': token_vocab,
        'metadata_vocab': metadata_vocab,
        'tokenizer': tokenizer,
        'token_metadata_counts': token_metadata_counts
    }
    torch.save(output, out_info_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Precompute Batches for LMC')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')

    # Model Hyperparameters
    parser.add_argument('--window', default=10, type=int)
    parser.add_argument('--metadata_samples', default=3, type=int)
    parser.add_argument('-bert', action='store_true', default=False)
    parser.add_argument('-sequential', action='store_true', default=False)
    parser.add_argument('--num_cpus', default=15, type=int)
    args = parser.parse_args()
    args.batch_size = 200 if args.bert else 1024
    args.metadata = 'section'
    print('Fixed batch size to {}'.format(args.batch_size))

    args.num_cpus = CPU_COUNT if args.num_cpus is None else min(CPU_COUNT, args.num_cpus)
    args.git_hash = get_git_revision_hash()
    render_args(args)
    loader = data_to_tens_bert if args.bert else data_to_tens
    precompute(args, loader)
