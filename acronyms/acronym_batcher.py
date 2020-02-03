from allennlp.data.tokenizers.token import Token
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

    def elmo_tokenize(self, tokens, vocab, indexer):
        tokens = list(map(lambda t: Token(t), tokens))
        return indexer.tokens_to_indices(tokens, vocab, 'elmo')['elmo']

    def elmo_next(self, vocab, indexer, sf_tokenized_lf_map):
        batch = self.batches[self.batch_ct]
        self.batch_ct += 1
        batch_size = batch.shape[0]
        target_lf_ids = np.zeros([batch_size, ], dtype=int)
        max_context_len = max([len(tt.split()) for tt in batch['trimmed_tokens'].tolist()])
        num_outputs = [len(sf_tokenized_lf_map[sf]) for sf in batch['sf'].tolist()]
        context_ids = np.zeros([batch_size, max_context_len, 50])
        max_output_length = max(num_outputs)
        max_lf_len = 5
        lf_ids = np.zeros([batch_size, max_output_length, max_lf_len, 50], dtype=int)
        for batch_idx, (_, row) in enumerate(batch.iterrows()):
            row = row.to_dict()
            sf = row['sf'].lower()
            # Find target_sf index in sf_lf_map
            target_lf_ids[batch_idx] = row['target_lf_idx']
            context_tokens = row['trimmed_tokens'].split()
            context_id_seq = self.elmo_tokenize(context_tokens, vocab, indexer)
            context_ids[batch_idx, :len(context_id_seq), :] = context_id_seq
            candidate_lfs = sf_tokenized_lf_map[row['sf']]
            for lf_idx, lf_toks in enumerate(candidate_lfs):
                lf_id_seq = self.elmo_tokenize(lf_toks, vocab, indexer)
                lf_ids[batch_idx, lf_idx, :len(lf_id_seq), :] = lf_id_seq

        return (context_ids, lf_ids, target_lf_ids), num_outputs

    def next(self, token_vocab, sf_lf_map, sf_tokenized_lf_map, lf_metadata_counts, metadata_vocab=None):
        batch = self.batches[self.batch_ct]
        self.batch_ct += 1
        batch_size = batch.shape[0]
        sf_ids = np.zeros([batch_size, ], dtype=int)
        section_ids = np.zeros([batch_size, ], dtype=int)
        category_ids = np.zeros([batch_size, ], dtype=int)
        num_contexts = np.zeros([batch_size, ], dtype=int)
        target_lf_ids = np.zeros([batch_size, ], dtype=int)
        max_context_len = max([len(tt.split()) for tt in batch['trimmed_tokens'].tolist()])
        num_outputs = [len(sf_tokenized_lf_map[sf]) for sf in batch['sf'].tolist()]
        context_ids = np.zeros([batch_size, max_context_len])
        max_output_length = max(num_outputs)
        max_lf_len = 5
        lf_ids = np.zeros([batch_size, max_output_length, max_lf_len], dtype=int)
        lf_token_ct = np.zeros([batch_size, max_output_length])
        max_num_metadata = metadata_vocab.size() if metadata_vocab is not None else 1
        actual_max_num_metadata = 0
        lf_metadata_p = np.zeros([batch_size, max_output_length, max_num_metadata])
        lf_metadata_ids = np.zeros([batch_size, max_output_length, max_num_metadata], dtype=int)
        m_vocab = metadata_vocab or token_vocab
        for batch_idx, (_, row) in enumerate(batch.iterrows()):
            row = row.to_dict()
            sf_ids[batch_idx] = token_vocab.get_id(row['sf'].lower())
            # Find target_sf index in sf_lf_map
            target_lf_ids[batch_idx] = row['target_lf_idx']
            context_id_seq = token_vocab.get_ids(row['trimmed_tokens'].split())
            num_context = len(context_id_seq)
            context_ids[batch_idx, :num_context] = context_id_seq
            num_contexts[batch_idx] = num_context
            candidate_lf_tokenized = sf_tokenized_lf_map[row['sf']]
            candidate_lf_senses = sf_lf_map[row['sf']]
            if 'section' in row:
                section_ids[batch_idx] = m_vocab.get_id(row['section'])
                if section_ids[batch_idx] < 0:
                    section_ids[batch_idx] = 0
                    # print('Not found={}'.format(row['section']))
            if 'category' in row:
                category_ids[batch_idx] = m_vocab.get_id(row['category'])
                if category_ids[batch_idx] < 0:
                    category_ids[batch_idx] = 0
                    # print('Not found={}'.format(row['category']))

            for lf_idx, lf_toks in enumerate(candidate_lf_tokenized):
                lf_id_seq = token_vocab.get_ids(lf_toks)
                lf_sense = candidate_lf_senses[lf_idx]
                assert len(lf_id_seq) <= max_lf_len
                num_toks = len(lf_id_seq)
                assert num_toks > 0
                lf_ids[batch_idx, lf_idx, :num_toks] = lf_id_seq
                lf_token_ct[batch_idx, lf_idx] = num_toks

                if lf_metadata_counts is not None:
                    if lf_sense in lf_metadata_counts:
                        lf_m = lf_metadata_counts[lf_sense]
                        k = 'section' if 'section' in lf_m.keys() else 'category'
                        lf_m_p, lf_m_name = lf_m['p'], lf_m[k]
                        metadata_ids = metadata_vocab.get_ids(lf_m_name)
                        num_m = len(lf_m_p)
                        actual_max_num_metadata = max(num_m, actual_max_num_metadata)
                        lf_metadata_p[batch_idx, lf_idx, :num_m] = lf_m_p
                        lf_metadata_ids[batch_idx, lf_idx, :num_m] = metadata_ids
                    else:
                        raise Exception('Cant find {} in lf metadata counts'.format(lf_sense))
                        # lf_metadata_p[batch_idx, lf_idx, 0] = 1

        lf_metadata_ids = lf_metadata_ids[:, :, :actual_max_num_metadata]
        lf_metadata_p = lf_metadata_p[:, :, :actual_max_num_metadata]

        return (sf_ids, section_ids, category_ids, context_ids, lf_ids, target_lf_ids, lf_token_ct,
                lf_metadata_ids), [lf_metadata_p], [num_outputs, num_contexts]

    def reset(self, shuffle=True):
        self.batch_ct = 0
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.batches = np.array_split(self.data, self.N // self.batch_size)
