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

    def get_prev_batch(self):
        return self.batches[self.batch_ct - 1]

    def next(self, vocab, sf_tokenized_lf_map):
        batch = self.batches[self.batch_ct]
        self.batch_ct += 1
        batch_size = batch.shape[0]
        sf_ids = np.zeros([batch_size, ], dtype=int)
        target_lf_ids = np.zeros([batch_size, ], dtype=int)
        max_context_len = max([len(tt.split()) for tt in batch['trimmed_tokens'].tolist()])
        max_global_len = max([len(tt) for tt in batch['tokenized_context_unique'].tolist()])
        num_outputs = [len(sf_tokenized_lf_map[sf]) for sf in batch['sf'].tolist()]
        context_ids = np.zeros([batch_size, max_context_len])
        global_ids = np.zeros([batch_size, max_global_len])
        max_output_length = max(num_outputs)
        max_lf_len = 5
        lf_ids = np.zeros([batch_size, max_output_length, max_lf_len], dtype=int)
        lf_token_ct = np.zeros([batch_size, max_output_length])
        global_token_ct = np.zeros([batch_size])
        for batch_idx, (_, row) in enumerate(batch.iterrows()):
            row = row.to_dict()
            sf_ids[batch_idx] = vocab.get_id(row['sf'].lower())
            # Find target_sf index in sf_lf_map
            target_lf_ids[batch_idx] = row['used_target_lf_idx']
            context_id_seq = vocab.get_ids(row['trimmed_tokens'].split())
            context_ids[batch_idx, :len(context_id_seq)] = context_id_seq
            candidate_lfs = sf_tokenized_lf_map[row['sf']]

            global_id_seq = vocab.get_ids(row['tokenized_context_unique'])
            num_global_ids = len(global_id_seq)
            global_ids[batch_idx, :num_global_ids] = global_id_seq
            global_token_ct[batch_idx] = num_global_ids

            for lf_idx, lf_toks in enumerate(candidate_lfs):
                lf_id_seq = vocab.get_ids(lf_toks)
                assert len(lf_id_seq) <= max_lf_len
                num_toks = len(lf_id_seq)
                assert num_toks > 0
                lf_ids[batch_idx, lf_idx, :num_toks] = lf_id_seq
                lf_token_ct[batch_idx, lf_idx] = num_toks

        return (sf_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct, global_ids, global_token_ct), num_outputs

    def reset(self, shuffle=True):
        self.batch_ct = 0
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.batches = np.array_split(self.data, self.N // self.batch_size)
