import numpy as np


class SkipGramBatchLoader:
    def __init__(self, N, section_idxs, batch_size=32):
        self.section_idxs = section_idxs
        self.batch_size = batch_size
        self.N = N
        self.batch_ct, self.batches = 0, None
        self.reset()

    def num_batches(self):
        return self.batches.shape[0]

    def has_next(self):
        return self.batch_ct < self.num_batches()

    def extract_context_ids(self, ids, center_idx, target_window):
        start_idx = max(0, center_idx - target_window)
        end_idx = min(len(ids), center_idx + target_window + 1)

        left_context = ids[start_idx:center_idx]
        right_context = ids[center_idx + 1:end_idx]

        section_boundary_left = np.where(left_context <= 0)[0]
        section_boundary_right = np.where(right_context <= 0)[0]

        left_trunc_idx = 0 if len(section_boundary_left) == 0 else section_boundary_left[-1] + 1
        right_trunc_idx = len(right_context) if len(section_boundary_right) == 0 else section_boundary_right[0]

        left_context_truncated = left_context[left_trunc_idx:]
        right_context_truncated = right_context[:right_trunc_idx]

        return np.concatenate([left_context_truncated, right_context_truncated])

    def _get_section_id_sample(self, token_section_samples, token_id):
        counter, sids = token_section_samples[token_id]
        sid = sids[counter]
        token_section_samples[token_id][0] += 1
        if token_section_samples[token_id][0] >= len(sids):
            np.random.shuffle(sids)
            token_section_samples[token_id] = [0, sids]
        return sid

    def next(self, ids, full_section_ids, token_section_samples, token_vocab, window_size):
        batch_idxs = self.batches[self.batch_ct]
        center_ids = ids[batch_idxs]
        context_ids = np.zeros([self.batch_size, (window_size * 2)], dtype=int)
        neg_ids = token_vocab.neg_sample(size=(self.batch_size, (window_size * 2)))

        center_section_ids = np.zeros([self.batch_size, ], dtype=int)
        context_section_ids = np.zeros([self.batch_size, (window_size * 2)], dtype=int)
        neg_section_ids = np.zeros([self.batch_size, (window_size * 2)], dtype=int)

        window_sizes = []
        for batch_idx, center_idx in enumerate(batch_idxs):
            example_context_ids = self.extract_context_ids(ids, center_idx, window_size)
            center_section_ids[batch_idx] = full_section_ids[center_idx]
            context_ids[batch_idx, :len(example_context_ids)] = example_context_ids
            window_sizes.append(len(example_context_ids))
            for idx, context_id in enumerate(example_context_ids):
                n_id = neg_ids[batch_idx, idx]
                c_sid = self._get_section_id_sample(token_section_samples, context_id)
                n_sid = self._get_section_id_sample(token_section_samples, n_id)
                context_section_ids[batch_idx, idx] = c_sid
                neg_section_ids[batch_idx, idx] = n_sid

        self.batch_ct += 1
        return (center_ids, center_section_ids, context_ids, context_section_ids, neg_ids, neg_section_ids,
                window_sizes)

    def reset(self):
        batch_idxs = np.array(list(set(np.arange(self.N)) - set(self.section_idxs)))
        num_batches = len(batch_idxs) // self.batch_size
        truncated_N = self.batch_size * num_batches
        np.random.shuffle(batch_idxs)
        self.batch_ct = 0
        self.batches = batch_idxs[:truncated_N].reshape(num_batches, self.batch_size)
