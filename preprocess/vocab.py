class Vocab:
    PAD_TOKEN = '<pad>'

    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.add_token(Vocab.PAD_TOKEN)

    def pad_id(self):
        return self.get_id(Vocab.PAD_TOKEN)

    def add_tokens(self, tokens):
        for tidx, token in enumerate(tokens):
            self.add_token(token)

    def add_token(self, token):
        if token not in self.w2i:
            self.w2i[token] = len(self.i2w)
            self.i2w.append(token)

    def get_id(self, token):
        if token in self.w2i:
            return self.w2i[token]
        return -1

    def get_ids(self, tokens):
        return list(map(self.get_id, tokens))

    def get_token(self, id):
        return self.i2w[id]

    def size(self):
        return len(self.i2w)
