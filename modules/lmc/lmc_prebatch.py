import itertools

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _get_metadata_id_sample(token_metadata_samples, token_id):
    """
    :param token_metadata_samples: dict for each token_id containing metadata samples drawn from p(metadata|token_id)
    :param token_id: key for token_metadata_samples
    :return: a list of sampled metadata ids drawn from empirical distribution p(metadata|token)

    We precompute Monte Carlo samples to avoid having to do it online within the main training script.
    All random samples are pre-computed before training and the sampling procedure merely involves selecting
    the next sample in token_metadata_samples[token_id].  This leads to a substantial speedup.
    """
    counter, sids = token_metadata_samples[token_id]
    sid = sids[counter]
    token_metadata_samples[token_id][0] += 1
    if token_metadata_samples[token_id][0] >= len(sids):
        token_metadata_samples[token_id] = [0, sids]
    return sid


class DistributedDataset(Dataset):
    """
    Wrapper over torch Dataset that constructs batch tensors for both regular LMC and experimental feature LMC BERT.
    Kwargs contains a data structure batches which groups the indices into ids.npy for all the center words into random
    batches.

    Kwargs contains the flattened ids file, vocabulary data structures necessary for converting tokens to ids,
    as well as empirical distributions / samples from p(metadata|token).
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getitem__(self, batch_ct):
        if self.kwargs['bert']:
            return self.get_bert_batch(batch_ct)
        else:
            return self.get_batch(batch_ct)

    def get_batch(self, batch_ct):
        """
        :param batch_ct: index into kwargs['batches'] for which to construct batch tensors
        :return: list of tensors representing context ids, center ids, negatively sampled ids,
        and metadata ids from MC sampling.
        """
        window_size = self.kwargs['window_size']
        ids = self.kwargs['ids']
        neg_sample_p = self.kwargs['neg_sample_p']
        full_metadata_ids = self.kwargs['full_metadata_ids']
        token_metadata_samples = self.kwargs['token_metadata_samples']

        batch_idxs = self.kwargs['batches'][batch_ct, :]
        batch_size = len(batch_idxs)

        neg_ids = np.random.choice(np.arange(len(neg_sample_p)), size=(batch_size, 2 * window_size), p=neg_sample_p)
        center_ids = ids[batch_idxs]
        context_ids = np.zeros([batch_size, (window_size * 2)], dtype=int)

        num_metadata = token_metadata_samples[1][1].shape[1]
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

        batch_data = (center_ids, center_metadata_ids, context_ids, context_metadata_ids, neg_ids, neg_metadata_ids,
                window_sizes)
        return list(map(torch.LongTensor, batch_data))

    def get_bert_batch(self, batch_ct):
        """
        :param batch_ct: index into kwargs['batches'] for which to construct batch tensors
        :return: list of tensors representing context ids, center ids, negatively sampled ids,
        and metadata ids from MC sampling.

        This is very similar to get_batch with the exception that all tokens and metadata are converted to ids
        not using unigram vocabularies: token_vocab and metadata_vocab but the wordpiece BerTokenizer.

        We construct BERT-style sequences with metadata in the following fashion:

        For context sequences (encoder): [
            [CLS] + <left_context> + <center_word> + <right_context> +
            [SEP] + <metadata> + <center_word> + [SEP]
        ]

        - The center word gets repeated with the metadata id after the [SEP] token to allow for deep joint modeling.

        In the LMC context words are represented as marginal distributions over metadata.  Given computational limits
        that are worsened when metadata vocabularies are large, we use Monte Carlo sampling from p(metadata|token) to
        approximate the full expectation.  Furthermore, given the large cost of running the Transformer, we pool all
        MC samples into a single sequence which allows for deep joint modeling between related metadata and a token.

        For both context words and negatively sampled tokens, we use the following format:

        [
            [CLS] + <token> + [SEP] + <MC metadata samples> + [SEP]
        ]

        We follow the standard protocol of assigning different token_type_ids to sequences to the left and right of the
        first [SEP] special token.
        """
        window_size = self.kwargs['window_size']
        ids = self.kwargs['ids']
        neg_sample_p = self.kwargs['neg_sample_p']
        full_metadata_ids = self.kwargs['full_metadata_ids']
        token_metadata_samples = self.kwargs['token_metadata_samples']
        wp_conversions = self.kwargs['wp_conversions']
        token_to_wp = wp_conversions['token_to_wp']
        meta_to_wp = wp_conversions['meta_to_wp']
        special_to_wp = wp_conversions['special_to_wp']

        PAD_ID = special_to_wp['[PAD]']
        CLS_ID = special_to_wp['[CLS]']
        SEP_ID = special_to_wp['[SEP]']

        batch_idxs = self.kwargs['batches'][batch_ct, :]
        batch_size = len(batch_idxs)
        center_ids = ids[batch_idxs]

        neg_ids = np.random.choice(np.arange(len(neg_sample_p)), size=(batch_size, 2 * window_size), p=neg_sample_p)

        num_metadata = token_metadata_samples[1][1].shape[-1]
        max_single_len = num_metadata + 2 + 5  # 2 for special tokens, 5 for max # of wps for unigram
        max_encoder_len = int((window_size * 2 + 1) * 2.5) + 2  # max avg of 2.5 wps ids per unigram in context window

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

            center_seq_bert = [CLS_ID] + center_tok_wp_ids + [SEP_ID] + center_meta_wp_ids
            center_encoded_len = min(len(center_seq_bert), max_single_len)
            center_bert_ids[batch_idx, :center_encoded_len] = center_seq_bert[:center_encoded_len]
            center_bert_mask[batch_idx, center_encoded_len:] = 0

            full_ids = left_context_ids + right_context_ids
            full_wp_ids = left_wp_ids + right_wp_ids

            flattened_left = list(itertools.chain(*left_wp_ids))
            flattened_right = list(itertools.chain(*right_wp_ids))
            c_wp_seq = flattened_left + center_tok_wp_ids + flattened_right
            right_seq = center_meta_wp_ids + center_tok_wp_ids
            full_context_wp_seq = [CLS_ID] + c_wp_seq + [SEP_ID] + right_seq + [SEP_ID]

            cutoff = min(len(full_context_wp_seq), max_encoder_len)
            context_bert_ids[batch_idx, :cutoff] = full_context_wp_seq[:cutoff]
            context_bert_mask[batch_idx, cutoff:] = 0
            left_cutoff = 2 + len(c_wp_seq)
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

        return batch_data

    def __len__(self):
        return len(self.kwargs['batches'])


def create_tokenizer_maps(bert_tokenizer, token_vocab, metadata_vocab):
    """
    :param bert_tokenizer: Pre-trained HuggingFace WordPiece tokenizer
    :param token_vocab: unigram token vocabulary for MIMIC-III
    :param metadata_vocab: metadata-specific vocabulary for MIMIC-III
    :return: dictionary mapping token_vocab and metadata_vocab ids to WordPiece ids

    This speeds up the batching process by only having to calculate WordPieces once for each unique word / metadata type
    """
    token_to_wp = [None] * token_vocab.size()
    meta_to_wp = [None] * metadata_vocab.size()

    for i in tqdm(range(token_vocab.size())):
        token = token_vocab.get_token(i)
        wps = bert_tokenizer.encode(token, add_special_tokens=False)
        token_to_wp[i] = wps
    for i in tqdm(range(metadata_vocab.size())):
        meta = metadata_vocab.get_token(i)
        wps = bert_tokenizer.encode(meta, add_special_tokens=False)
        meta_to_wp[i] = wps

    special_map = {}
    for t, i in zip(bert_tokenizer.all_special_tokens, bert_tokenizer.all_special_ids):
        if 'header' not in t and 'document' not in t:
            special_map[t] = i
    return {'token_to_wp': token_to_wp, 'meta_to_wp': meta_to_wp, 'special_to_wp': special_map}


def extract_context_ids(ids, center_idx, target_window):
    """
    :param ids: Flattened list of all token and metadata ids
    :param center_idx: Index into ids from which to extract the center id
    :param target_window: Distance to the left and right of center id for which to extract context
    :return: Sequence of ids representing [left_context] + [center_id] + [right_context]

    We truncate sequences when a metadata or document boundary exists before the target window is reached.
    """
    l, r = extract_full_context_ids(ids, center_idx, target_window)
    return np.concatenate([l, r])


def extract_full_context_ids(ids, center_idx, target_window):
    """
    :param ids: Flattened list of all token and metadata ids
    :param center_idx: Index into ids from which to extract the center id
    :param target_window: Distance to the left and right of center id for which to extract context
    :return: Tuple of left context, right context, which are just lists of ids surrounding ids[center_idx]

    We truncate sequences when a metadata or document boundary exists before the target window is reached.
    """
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


def generate_metadata_samples(token_metadata_counts, metadata_vocab, sample=5):
    """
    :param token_metadata_counts: dict for each token_id containing empirical counts for p(metadata|token_id)
    :param metadata_vocab: vocabulary for metadata (necessary just for size of metadata vocabulary)
    :param sample: Number of Monte Carlo samples to make
    :return: dict for each token_id containing metadata samples drawn from p(metadata|token_id)

    We precompute Monte Carlo samples to avoid having to do it online within the main training script.
    All random samples are pre-computed before training and the sampling procedure merely involves selecting
    the next sample in token_metadata_samples[token_id].  This leads to a substantial speedup.
    """
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
