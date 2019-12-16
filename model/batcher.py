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

    def next(self, ids, window_size, add_section_as_context=False):
        batch_idxs = self.batches[self.batch_ct]
        center_ids = ids[batch_idxs]
        num_pseudo_contexts = 1 if add_section_as_context else 0
        context_ids = np.zeros([self.batch_size, (window_size * 2) + num_pseudo_contexts], dtype=int)
        actual_window_sizes = []
        for batch_idx, center_idx in enumerate(batch_idxs):
            example_context_ids = self.extract_context_ids(ids, center_idx, window_size)
            actual_window_size = len(example_context_ids) + num_pseudo_contexts
            if add_section_as_context:
                section_idx = center_idx - 1
                while ids[section_idx] > 0:
                    section_idx -= 1
                section_id = ids[section_idx]
                assert section_id <= 0
                context_ids[batch_idx, 0] = - section_id
            context_ids[batch_idx, num_pseudo_contexts:actual_window_size] = example_context_ids
            actual_window_sizes.append(actual_window_size)
        self.batch_ct += 1
        return center_ids, context_ids, actual_window_sizes

    def reset(self):
        batch_idxs = np.array(list(set(np.arange(self.N)) - set(self.section_idxs)))
        num_batches = len(batch_idxs) // self.batch_size
        truncated_N = self.batch_size * num_batches
        np.random.shuffle(batch_idxs)
        self.batch_ct = 0
        self.batches = batch_idxs[:truncated_N].reshape(num_batches, self.batch_size)
