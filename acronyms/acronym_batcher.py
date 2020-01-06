import numpy as np


class AcronymBatcherLoader:
    def __init__(self, df, batch_size=32):
        self.batch_size = batch_size
        self.N = df.shape[0]
        self.data = df
        self.batch_ct, self.batches = 0, None

    def num_batches(self):
        return len(self.batches)

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

    def get_prev_batch(self):
        return self.batches[self.batch_ct - 1]

    def next(self, vocab, sf_tokenized_lf_map):
        batch = self.batches[self.batch_ct]
        self.batch_ct += 1
        batch_size = batch.shape[0]
        sf_ids = np.zeros([batch_size, ], dtype=int)
        target_lf_ids = np.zeros([batch_size, ], dtype=int)
        max_context_len = max([len(tt.split()) for tt in batch['trimmed_tokens'].tolist()])
        num_outputs = [len(sf_tokenized_lf_map[sf]) for sf in batch['sf'].tolist()]
        context_ids = np.zeros([batch_size, max_context_len])
        max_output_length = max(num_outputs)
        max_lf_len = 5
        lf_ids = np.zeros([batch_size, max_output_length, max_lf_len], dtype=int)
        lf_token_ct = np.zeros([batch_size, max_output_length])
        max_lf_token_ct = 0
        for batch_idx, (_, row) in enumerate(batch.iterrows()):
            row = row.to_dict()
            sf_ids[batch_idx] = vocab.get_id(row['sf'].lower())
            # Find target_sf index in sf_lf_map
            target_lf_ids[batch_idx] = row['used_target_lf_idx']
            context_id_seq = vocab.get_ids(row['trimmed_tokens'].split())
            context_ids[batch_idx, :len(context_id_seq)] = context_id_seq
            candidate_lfs = sf_tokenized_lf_map[row['sf']]
            for lf_idx, lf_toks in enumerate(candidate_lfs):
                lf_id_seq = vocab.get_ids(lf_toks)
                lf_id_seq_trunc = lf_id_seq[:min(max_lf_len, len(lf_id_seq))]
                num_toks = len(lf_id_seq_trunc)
                assert num_toks > 0
                lf_ids[batch_idx, lf_idx, :num_toks] = lf_id_seq_trunc
                lf_token_ct[batch_idx, lf_idx] = num_toks
                max_lf_token_ct = max(max_lf_token_ct, num_toks)

        return (sf_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct), num_outputs

    def reset(self, shuffle=True):
        self.batch_ct = 0
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.batches = np.array_split(self.data, self.N // self.batch_size)
