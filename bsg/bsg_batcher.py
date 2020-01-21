import numpy as np


class SkipGramBatchLoader:
    def __init__(self, N, metadata_idxs, batch_size=32):
        self.metadata_idxs = metadata_idxs
        self.batch_size = batch_size
        self.N = N
        self.batch_ct, self.batches = 0, None
        self.reset()
        self.cached_boundaries = None

    def num_batches(self):
        return self.batches.shape[0]

    def has_next(self):
        return self.batch_ct < self.num_batches()

    def get_boundary_idxs(self, ids, center_idx, target_window):
        if ids[center_idx] <= 0:
            return None, None

        lower_bound = max(1, center_idx - target_window)
        for left_boundary in range(center_idx, lower_bound - 1, -1):
            if ids[left_boundary - 1] == -1:
                break

        upper_bound = min(center_idx + 1 + target_window, len(ids))
        for right_boundary in range(center_idx + 1, upper_bound + 1):
            if right_boundary == len(ids) or ids[right_boundary] == -1:
                break

        return left_boundary, right_boundary

    def fill_boundaries(self, ids, target_window):
        print('Caching windows for each positional idx...')
        self.cached_boundaries = list(map(
            lambda center_idx: self.get_boundary_idxs(ids, center_idx, target_window), range(len(ids))))

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

    def extract_cached_context_ids(self, ids, center_idx, target_window):
        if self.cached_boundaries is None:
            self.fill_boundaries(ids, target_window)

        left_idx, right_idx = self.cached_boundaries[center_idx]
        left_context = ids[left_idx:center_idx]
        right_context = ids[center_idx + 1: right_idx]
        full_context = np.concatenate([left_context, right_context])
        return full_context

    def next(self, ids, full_sec_ids, full_cat_ids, vocab, window_size, use_cache=False):
        batch_idxs = self.batches[self.batch_ct]
        center_ids = ids[batch_idxs]
        context_ids = np.zeros([self.batch_size, (window_size * 2)], dtype=int)
        window_sizes = []

        neg_ids = vocab.neg_sample(size=(self.batch_size, (window_size * 2)))
        sec_ids = np.zeros([self.batch_size,], dtype=int)
        cat_ids = np.zeros([self.batch_size,], dtype=int)

        extractor = self.extract_cached_context_ids if use_cache else self.extract_context_ids
        for batch_idx, center_idx in enumerate(batch_idxs):
            sec_ids[batch_idx] = full_sec_ids[center_idx]
            cat_ids[batch_idx] = full_cat_ids[center_idx]
            example_context_ids = extractor(ids, center_idx, window_size)
            actual_window_size = len(example_context_ids)
            context_ids[batch_idx, :actual_window_size] = example_context_ids
            window_sizes.append(actual_window_size)
        self.batch_ct += 1
        return center_ids, sec_ids, cat_ids, context_ids, neg_ids, window_sizes

    def reset(self):
        batch_idxs = np.array(list(set(np.arange(self.N)) - set(self.metadata_idxs)))
        num_batches = len(batch_idxs) // self.batch_size
        truncated_N = self.batch_size * num_batches
        np.random.shuffle(batch_idxs)
        self.batch_ct = 0
        self.batches = batch_idxs[:truncated_N].reshape(num_batches, self.batch_size)
