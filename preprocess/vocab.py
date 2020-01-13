import numpy as np


class Vocab:
    PAD_TOKEN = '<pad>'

    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.support = []
        self.add_token(Vocab.PAD_TOKEN)
        self.cached_neg_sample_prob = None
        self.separator_start_vocab_id = None

    def pad_id(self):
        return self.get_id(Vocab.PAD_TOKEN)

    def add_tokens(self, tokens, token_support=1):
        for tidx, token in enumerate(tokens):
            self.add_token(token, token_support=token_support)

    def add_token(self, token, token_support=1):
        if token not in self.w2i:
            self.w2i[token] = len(self.i2w)
            self.i2w.append(token)
            self.support.append(0)
        self.support[self.get_id(token)] += token_support
        return self.w2i[token]

    def neg_sample(self, size=None):
        if self.cached_neg_sample_prob is None:
            support = np.array(self.support)
            support_raised = np.power(support, 0.75)
            support_raised[0] = 0.0  # Never select padding idx
            self.cached_neg_sample_prob = support_raised / support_raised.sum()
        return np.random.choice(np.arange(self.size()), size=size, p=self.cached_neg_sample_prob)

    def get_id(self, token):
        if token in self.w2i:
            return self.w2i[token]
        return -1

    def id_count(self, id):
        return self.support[id]

    def token_count(self, token):
        return self.id_count(self.get_id(token))

    def truncate(self):
        print('Removing section pseudo-tokens from vocabulary...')
        self.support = self.support[:self.separator_start_vocab_id]
        self.i2w = self.i2w[:self.separator_start_vocab_id]

    def get_ids(self, tokens):
        return list(map(self.get_id, tokens))

    def get_token(self, id):
        return self.i2w[id]

    def size(self):
        return len(self.i2w)
