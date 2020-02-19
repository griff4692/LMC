import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from transformers import AlbertConfig, AlbertModel


class LMCDecoderBERT(nn.Module):
    def __init__(self, args, token_vocab_size, output_dim=200):
        super(LMCDecoderBERT, self).__init__()
        self.pool_layers = True

        if args.debug_model:
            bert_dim = 100
            num_hidden_layers = 1
            embedding_size = 100
            intermediate_size = 100
            output_dim = 100
        else:
            bert_dim = 256
            num_hidden_layers = 2
            embedding_size = 128
            intermediate_size = 256
        num_attention_heads = max(1, bert_dim // 64)
        print('Using {} attention heads in decoder'.format(num_attention_heads))

        config = AlbertConfig(
            vocab_size=token_vocab_size,
            embedding_size=embedding_size,
            hidden_size=bert_dim,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,  # 3072 is default
            num_attention_heads=num_attention_heads,
            output_hidden_states=self.pool_layers
        )

        self.bert = AlbertModel(config)

        self.u = nn.Linear(bert_dim, output_dim, bias=True)
        self.v = nn.Linear(bert_dim, 1, bias=True)

    def forward(self, **kwargs):
        if self.pool_layers:
            all_encoded_layers = self.bert(**kwargs)[2]
            num_layers = len(all_encoded_layers)
            last_layers = min(num_layers - 1, 4)
            last_encoded_layers = torch.stack(all_encoded_layers[-last_layers:]).sum(0)
            att_mask = kwargs['attention_mask']
            denom = att_mask.sum(1).unsqueeze(-1)
            last_encoded_layers *= att_mask.unsqueeze(-1)
            h = last_encoded_layers.sum(1) / denom
        else:
            output = self.bert(**kwargs)[0]
            att_mask = kwargs['attention_mask']
            denom = att_mask.sum(1).unsqueeze(-1)
            output *= att_mask.unsqueeze(-1)
            h = output.sum(1) / denom

        return self.u(h), self.v(h).exp()


class LMCDecoder(nn.Module):
    def __init__(self, token_vocab_size, metadata_vocab_size, input_dim=100, hidden_dim=64, output_dim=100):
        super(LMCDecoder, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.f = nn.Linear(input_dim * 2, hidden_dim, bias=True)
        self.u = nn.Linear(hidden_dim, output_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=True)

        self.token_embeddings = nn.Embedding(token_vocab_size, input_dim, padding_idx=0)
        self.metadata_embeddings = nn.Embedding(metadata_vocab_size, input_dim, padding_idx=0)

    def forward(self, center_ids, metadata_ids, normalizer=None):
        """
        :param center_ids: LongTensor of batch_size
        :param metadata_ids: LongTensor of batch_size
        :return: mu (batch_size, latent_dim), var (batch_size, 1)
        """
        center_embedding = self.token_embeddings(center_ids)
        if len(center_ids.size()) > len(metadata_ids.size()):
            center_embedding = center_embedding.sum(1) / normalizer

        metadata_embedding = self.metadata_embeddings(metadata_ids)
        merged_embeds = self.dropout(torch.cat([center_embedding, metadata_embedding], dim=-1))
        h = self.dropout(F.relu(self.f(merged_embeds)))
        return self.u(h), self.v(h).exp()
