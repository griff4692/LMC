import numpy as np


class BSGBatchLoader:
    """
    Shuffles center words and converts into batched tensors online
    """
    def __init__(self, N, metadata_idxs, batch_size=1024):
        """
        :param N: total corpus length as measured in tokens (inclusive of all metadata pseudo tokens
        even though they are notmodeled as the center word)
        :param metadata_idxs: Positions in ids array which represented metadata tokens
        (these are removed as center words)
        :param batch_size: training batch size
        """
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

    def extract_context_ids(self, ids, center_idx, target_window):
        """
        :param ids: Flattened list of all token and metadata ids
        :param center_idx: Index into ids from which to extract the center id
        :param target_window: Distance to the left and right of center id for which to extract context
        :return: Sequence of ids representing [left_context] + [right_context]

        We truncate sequences when a metadata or document boundary exists before the target window is reached.
        """
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

    def next(self, ids, full_meta_ids, vocab, window_size):
        """
        :param ids: Flattened list of all token and metadata ids
        :param full_meta_ids: For each element in ids provides its corresponding metadata id
        :param vocab: token vocabulary from which we negatively sample words
        :param window_size: Distance to the left and right of center word for which to extract context
        :return:
        """
        batch_idxs = self.batches[self.batch_ct]
        center_ids = ids[batch_idxs]
        context_ids = np.zeros([self.batch_size, (window_size * 2)], dtype=int)
        window_sizes = []

        neg_ids = vocab.neg_sample(size=(self.batch_size, (window_size * 2)))
        meta_ids = np.zeros([self.batch_size,], dtype=int)
        for batch_idx, center_idx in enumerate(batch_idxs):
            meta_ids[batch_idx] = full_meta_ids[center_idx]
            example_context_ids = self.extract_context_ids(ids, center_idx, window_size)
            actual_window_size = len(example_context_ids)
            context_ids[batch_idx, :actual_window_size] = example_context_ids
            window_sizes.append(actual_window_size)
        self.batch_ct += 1
        return center_ids, meta_ids, context_ids, neg_ids, window_sizes

    def reset(self):
        """
        :return: None

        Each index in ids of length N represents a training example (center word)
        after removing metadata tokens from consideration.
        In this function, we merely randomize the order in which these center words are trained and group into batches.
        """
        batch_idxs = np.array(list(set(np.arange(self.N)) - set(self.metadata_idxs)))
        num_batches = len(batch_idxs) // self.batch_size
        truncated_N = self.batch_size * num_batches
        np.random.shuffle(batch_idxs)
        self.batch_ct = 0
        self.batches = batch_idxs[:truncated_N].reshape(num_batches, self.batch_size)
