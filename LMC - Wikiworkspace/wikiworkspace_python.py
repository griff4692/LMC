#!/usr/bin/env python
# coding: utf-8

# In[92]:


import glob,os
import sys
import pandas as pd
import string
import numpy as np
from tqdm import tqdm
from time import sleep
import sys
from gensim.corpora import WikiCorpus
from multiprocessing import Pool
import json
import pickle
from collections import defaultdict
import gc
import torch
from scipy import sparse
from scipy.sparse import dok_matrix
import random


# In[2]:


class Vocab:
    PAD_TOKEN = '<pad>'

    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.support = []
        self.add_token(Vocab.PAD_TOKEN)
        self.cached_neg_sample_prob = None

    def pad_id(self):
        return self.get_id(Vocab.PAD_TOKEN)

    def add_tokens(self, tokens):
        for tidx, token in enumerate(tokens):
            self.add_token(token)

    def add_token(self, token, token_support=1):
        if token not in self.w2i:
            self.w2i[token] = len(self.i2w)
            self.i2w.append(token)
            self.support.append(0)
        self.support[self.get_id(token)] += token_support

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

    def get_ids(self, tokens):
        return list(map(self.get_id, tokens))

    def get_token(self, id):
        return self.i2w[id]

    def size(self):
        return len(self.i2w)


# In[17]:


class extract_data:

    def __init__(self,data_path,path,keep_prob):
        self.X = []
        self.vocab = []
        self.tokens = []
        self.keep_prob = []
        self.subsampled_ids = []
        self.tokenized_subsampled_data = []
        self.data_path = data_path
        self.path = path
        self.keep_prob = keep_prob
        
    def make_corpus(self,in_f, out_f):
        print("Loading data")
        wiki = WikiCorpus(in_f)
        print("Processing...")
        i = 1
        for document in wiki.get_texts():
            out_f_ = ''.join([out_f,'\\wiki_en_',str(i),".txt"])
            output = open(out_f_, 'w',encoding='utf-8')
            output.write(' '.join(document))
            output.close()
            i+=1
            if i%10000 == 0:  
                print(i,"documents procecessed")
        print('Processing complete!')

    def read_data(self):
            X = []
            keep_prob = self.keep_prob
            print("Reading txt data")
            os.chdir(self.data_path)
            N = len(glob.glob("*.txt"))
            #choice = np.random.choice(N, sample_size, replace=False)+1
            for i in tqdm(range(1,N+1), position=0, leave=True):
                if np.random.binomial(1, keep_prob):
                    text = ''.join([self.data_path,"\\wiki_en_",str(i),".txt"])
                    with open(text,'r',encoding='utf-8') as f:
                        X.append(''.join(list(f)).lower())
                else:
                    pass
            print("Saving sampled wiki data, sample rate:",keep_prob)
            with open(self.path + 'Wiki_tokenized{}.json'.format(''), 'w') as fd:
                json.dump(list(zip(range(len(X)),X)), fd)
    
    def token_counts(self):
        print("Generating token counts")
        token_cts = defaultdict(int)
        doc_id = 0
        token_counts_fn = self.path + 'Wiki_tokenized{}.json'.format('')
        with open(token_counts_fn, 'r') as fd:
            X = json.load(fd)
        for doc in X:
            for token in doc[1].split():
                token_cts[token] += 1
                token_cts['__ALL__'] += 1
        print("Saving token counts")
        with open(self.path + 'Wiki_token_counts{}.json'.format(''), 'w') as fd:
            json.dump(token_cts, fd)
        #with open((self.path +'Wiki_doc_counts{}.json').format(''),w) as fd:
        #    json.dump(doc_cts,fd)
    
    def subsample(self):
        print("Sub-sampling tokens...")
        tokenized_fp = self.path + 'Wiki_tokenized'
        token_counts_fp = self.path + 'Wiki_token_counts'
        subsample_param = 0.001
        min_token_count = 5
        debug_str = ''
        tokenized_data_fn = '{}{}.json'.format(tokenized_fp, debug_str)
        with open(tokenized_data_fn, 'r') as fd:
            tokenized_data = json.load(fd)
        token_counts_fn = '{}{}.json'.format(token_counts_fp, debug_str)
        with open(token_counts_fn, 'r') as fd:
            token_counts = json.load(fd)
        N = float(token_counts['__ALL__'])
        # And vocabulary with word counts
        self.vocab = Vocab()
        num_docs = len(tokenized_data)
        for doc_idx in tqdm(range(num_docs), position=0, leave=True):
            category, tokenized_doc_str = tokenized_data[doc_idx]
            subsampled_doc = []
            for token in tokenized_doc_str.split():
                wc = token_counts[token]
                too_sparse = wc <= min_token_count
                if too_sparse:
                    continue
                frac = wc / N
                keep_prob = min((np.sqrt(frac / subsample_param) + 1) * (subsample_param / frac), 1.0)
                should_keep = np.random.binomial(1, keep_prob) == 1
                if should_keep:
                    subsampled_doc.append(token)
                    self.vocab.add_token(token, token_support=1)
            self.tokenized_subsampled_data.append((category-1, ' '.join(subsampled_doc)))
            
    def tokens_to_ids(self):
        print("Converting tokens to ids...")
        for i in tqdm(range(len(self.tokenized_subsampled_data)), position=0, leave=True):
            self.subsampled_ids.append(np.asarray([self.vocab.get_id(x) for x in self.tokenized_subsampled_data[i][1].split()]))
        self.subsampled_ids = np.asarray(self.subsampled_ids)
    
    def token_doc_map(self):
        print("Forming token document matrix... ")
        self.token_doc_matrix = dok_matrix((self.vocab.size(),self.subsampled_ids.shape[0]), dtype=np.int16)
        for i in tqdm(range(self.subsampled_ids.shape[0]), position=0, leave=True):
            for token_id in self.subsampled_ids[i]:
                self.token_doc_matrix[token_id,i] +=1


# In[34]:


in_f = "D:\Latent Meaning Cells\simplewiki-latest-pages-articles.xml.bz2"
out_f = 'D:\Latent Meaning Cells\simplewiki'

data_path =  "D:\\Latent Meaning Cells\\simplewiki" #path where you read txt files
path = "D:\\Latent Meaning Cells\\" #path where you output json files

wiki_data = extract_data(data_path,path,keep_prob = 0.01)
#wiki_data.make_corpus(in_f, out_f)
try:
    p = Pool(processes=10)
    p.apply(wiki_data.read_data()) #run if you want to read txt files
except:
    p.close()
    wiki_data.token_counts() #if you already have json files start from here
    wiki_data.subsample()
    wiki_data.tokens_to_ids()
    wiki_data.token_doc_map()


# In[35]:


sub_tokens = wiki_data.subsampled_ids
vocab = wiki_data.vocab
token_doc_matrix = wiki_data.token_doc_matrix


# In[58]:


doc_cts = defaultdict(list)
for i in range(sub_tokens.shape[0]):
    [doc_cts[x].append(i) for x in sub_tokens[i]]
for key in doc_cts.keys():
    doc_cts[key] = np.asarray(doc_cts[key])


# In[93]:


perm_token_doc = []
for i in tqdm(range(sub_tokens.shape[0])):
    perm_token_doc += [(x,i) for x in range(sub_tokens[i].shape[0])]
random.shuffle(perm_token_doc)
perm_token_doc = np.asarray(perm_token_doc)


# In[6]:


#filename = "D:\Latent Meaning Cells\\vocab.obj" 
#file_pi = open(filename, 'wb')
#pickle.dump(vocab, file_pi)


# If already extracted:

# In[7]:


#filename = "D:\Latent Meaning Cells\sub_tokens.obj"
#filehandler = open(filename, 'rb')
#sub_tokens = pickle.load(filehandler)


# In[8]:


#filename = "D:\Latent Meaning Cells\\vocab.obj"
#filehandler = open(filename, 'rb')
#vocab = pickle.load(filehandler)


# In[ ]:


def neg_doc_sample(vocab.neg_sample(10)):
    [neg_doc_sample(x) for x in vocab.neg_sample(10)]


# In[119]:


class batcher:

    def __init__(self,batch_size,window_size,vocab,sub_tokens,doc_cts,perm_token_doc):

        
        self.batch_size = batch_size
        self.window_size = window_size
        self.vocab = vocab
        self.sub_tokens = sub_tokens
        self.doc_cts = doc_cts
        self.vocab_tokens = np.linspace(1, vocab.size()-1, num=vocab.size()-1).astype(int)
        #self.prob = np.power(vocab.support[1:], 0.75)
        #self.prob = self.prob/np.sum(self.prob)
        self.layer = 0
        self.perm_token_doc = perm_token_doc
        
    def next(self):
        
        layer = self.layer
        
        perm_token_doc = self.perm_token_doc
        sub_tokens = self.sub_tokens
        batch_size = min(self.batch_size,perm_token_doc[self.batch_size*layer:self.batch_size*(layer+1),1].shape[0])
        window_size = self.window_size
        #prob = self.prob
        
        center_words = np.zeros(batch_size)
        vocab_tokens =self.vocab_tokens
        num_contexts = np.zeros(batch_size)
        
        positive_words = np.zeros((batch_size,window_size*2))
        negative_words = np.zeros((batch_size,window_size*2))
        negative_docs = np.zeros((batch_size,window_size*2))
        #doc_ids = np.random.choice(len(sub_tokens),batch_size)
        
        doc_ids = perm_token_doc[batch_size*layer:batch_size*(layer+1),1]
        center_index = perm_token_doc[batch_size*layer:batch_size*(layer+1),0]
        
        len_docs = np.asarray([x.shape[0] for x in sub_tokens[doc_ids]])
        #center_index = np.asarray([np.random.choice(x) for x in len_docs])
        upper_index = np.minimum(center_index+window_size,len_docs-1).astype(int)
        lower_index = np.maximum(center_index-window_size,np.zeros(batch_size)).astype(int)
        
        if batch_size<self.batch_size:
            self.layer = 0
        else:
            self.layer +=1 
        
        
        for i in range(batch_size):
        
            positive_sub_batch = np.linspace(lower_index[i],upper_index[i], num=upper_index[i]-lower_index[i]+1)
            positive_sub_batch = positive_sub_batch[positive_sub_batch != center_index[i]].astype(int)
            
            num_contexts[i] = positive_sub_batch.shape[0]
            
            document = sub_tokens[doc_ids[i]]
            positive_sub_batch = np.asarray([document[x] for x in positive_sub_batch]).astype(int)
            positive_words[i,:positive_sub_batch.shape[0]] = positive_sub_batch

            center_words[i] = document[center_index[i]]
            
            #negative_words_ = vocab_tokens[~np.isin(vocab_tokens, positive_sub_batch)]
            #negative_sampling_probability = prob[~np.isin(vocab_tokens, positive_sub_batch)]
            #negative_sampling_probability = negative_sampling_probability/np.sum(negative_sampling_probability)
            #negative_words[i] = np.random.choice(negative_words_, window_size*2, p=negative_sampling_probability).astype(int)
            
            negative_words[i] = vocab.neg_sample(window_size*2)
            negative_docs[i] = np.asarray([np.random.choice(doc_cts[x]) for x in negative_words[i]])
            
        return doc_ids.astype(int), center_words.astype(int), positive_words.astype(int), negative_words.astype(int),num_contexts.astype(int), negative_docs.astype(int)


# In[120]:


def mask_2D(target_size, num_contexts):
    mask = torch.BoolTensor(target_size)
    mask.fill_(0)
    for batch_idx, num_c in enumerate(num_contexts):
        if num_c < target_size[1]:
            mask[batch_idx, num_c:] = 1
    return mask


# In[121]:


import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, device,encoder_input_dim,encoder_hidden_dim,latent_dim, token_vocab_size, section_vocab_size):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(encoder_input_dim,encoder_hidden_dim,latent_dim, token_vocab_size, section_vocab_size)
        self.margin = 1.0

    def forward(self, center_ids, section_ids, context_ids, neg_context_ids,neg_section_ids,num_context_ids):
        """
        :param center_ids: batch_size
        :param section_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :param neg_context_ids: batch_size, 2 * context_window
        :param num_contexts: batch_size (how many context words for each center id - necessary for masking padding)
        :return: cost components: KL-Divergence (q(z|w,c) || p(z|w)) and max margin (reconstruction error)
        """
        # Mask padded context ids
        batch_size, num_context_ids = context_ids.size()
        mask_size = torch.Size([batch_size, num_context_ids])
        mask = mask_2D(mask_size, num_contexts).to(self.device)

        # Compute center words
        mu_center, sigma_center = self.encoder(center_ids, section_ids)
        mu_center_tiled = mu_center.unsqueeze(1).repeat(1, num_context_ids, 1)
        sigma_center_tiled = sigma_center.unsqueeze(1).repeat(1, num_context_ids, 1)
        mu_center_flat = mu_center_tiled.view(batch_size * num_context_ids, -1)
        sigma_center_flat = sigma_center_tiled.view(batch_size * num_context_ids, -1)

        # Tile section ids for positive and negative samples
        section_ids_tiled = section_ids.unsqueeze(-1).repeat(1, num_context_ids)

        # Compute positive and negative encoded samples
        mu_pos_context, sigma_pos_context = self.encoder(context_ids, section_ids_tiled)
        mu_neg_context, sigma_neg_context = self.encoder(neg_context_ids, neg_section_ids)
        
        # Flatten positive context
        mu_pos_context_flat = mu_pos_context.view(batch_size * num_context_ids, -1)
        sigma_pos_context_flat = sigma_pos_context.view(batch_size * num_context_ids, -1)

        # Flatten negative context
        mu_neg_context_flat = mu_neg_context.view(batch_size * num_context_ids, -1)
        sigma_neg_context_flat = sigma_neg_context.view(batch_size * num_context_ids, -1)

        # Compute KL-divergence between center words and negative and reshape
        kl_pos_flat = compute_kl(mu_center_flat, sigma_center_flat, mu_pos_context_flat, sigma_pos_context_flat)
        kl_neg_flat = compute_kl(mu_center_flat, sigma_center_flat, mu_neg_context_flat, sigma_neg_context_flat)
        kl_pos = kl_pos_flat.view(batch_size, num_context_ids)
        kl_neg = kl_neg_flat.view(batch_size, num_context_ids)

        hinge_loss = (kl_pos - kl_neg + self.margin).clamp_min_(0)
        hinge_loss = hinge_loss.masked_fill(mask, 0)
        hinge_loss = hinge_loss.sum(1)
        return hinge_loss.mean()


# In[122]:


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
 
class Encoder(nn.Module):
    def __init__(self, encoder_input_dim,encoder_hidden_dim,latent_dim, token_vocab_size, section_vocab_size):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.f = nn.Linear(encoder_input_dim * 2, encoder_hidden_dim, bias=True)
        self.u = nn.Linear(encoder_hidden_dim, latent_dim, bias=True)
        self.v = nn.Linear(encoder_hidden_dim, 1, bias=True)
    
        self.token_embeddings = nn.Embedding(token_vocab_size, encoder_input_dim, padding_idx=0)
        self.token_embeddings.weight.data.uniform_(-2, 2)
        self.section_embeddings = nn.Embedding(section_vocab_size, encoder_input_dim)
        self.section_embeddings.weight.data.uniform_(-2, 2)
        
    def forward(self, center_ids, section_ids):
        """
        :param center_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), logvar (batch_size, 1)
        """
        center_embedding = self.token_embeddings(center_ids)
        section_embedding = self.section_embeddings(section_ids)
            
        merged_embeds = self.dropout(torch.cat([center_embedding, section_embedding], dim=-1))
            
        h = self.dropout(F.relu(self.f(merged_embeds)))
        var_clamped = self.v(h).exp().clamp_min(1.0)
        return self.u(h), var_clamped


# In[123]:


def compute_kl(mu_a, sigma_a, mu_b, sigma_b, device=None):
    """
    :param mu_a: mean vector of batch_size x dim
    :param sigma_a: standard deviation of batch_size x {1, dim}
    :param mu_b: mean vector of batch_size x dim
    :param sigma_b: standard deviation of batch_size x {1, dim}
    :return: computes KL-Divergence between 2 diagonal Gaussian (a||b)
    """
    var_dim = sigma_a.size()[-1]
    assert sigma_b.size()[-1] == var_dim
    if var_dim == 1:
        return kl_spher(mu_a, sigma_a, mu_b, sigma_b)
    return kl_diag(mu_a, sigma_a, mu_b, sigma_b, device=device)


# In[124]:


def kl_spher(mu_a, sigma_a, mu_b, sigma_b):
    """
    :param mu_a: mean vector of batch_size x dim
    :param sigma_a: standard deviation of batch_size x 1
    :param mu_b: mean vector of batch_size x dim
    :param sigma_b: standard deviation of batch_size x 1
    :return: computes KL-Divergence between 2 spherical Gaussian (a||b)
    """
    d = mu_a.shape[1]
    sigma_p_inv = 1.0 / sigma_b  # because diagonal
    tra = d * sigma_a * sigma_p_inv
    quadr = sigma_p_inv * torch.pow(mu_b - mu_a, 2).sum(1, keepdim=True)
    log_det = - d * torch.log(sigma_a * sigma_p_inv)
    res = 0.5 * (tra + quadr - d + log_det)
    return res


# In[125]:


device="cuda"
encoder_input_dim = 100
encoder_hidden_dim = 64
latent_dim = 100

token_vocab_size = vocab.size()
section_vocab_size = sub_tokens.shape[0]

model = VAE(device,encoder_input_dim,encoder_hidden_dim,latent_dim, token_vocab_size, section_vocab_size).to(device)

trainable_params = filter(lambda x: x.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=0.01)
optimizer.zero_grad()


# In[16]:


model.load_state_dict(torch.load( "D:\\Latent Meaning Cells\\checkpoint.pth"))


# In[ ]:


window_size = 5
batch_size = 1024
num_epoch = 300
num_contexts = batch_size
generator = batcher(batch_size,window_size,vocab,sub_tokens,doc_cts,perm_token_doc)


for epoch in range(1, num_epoch + 1):
    sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
    print('Starting Epoch={}'.format(epoch))
    num_batches = batch_size
    loss_array = []
    epoc_loss = 0
    for _ in tqdm(range(int(perm_token_doc.shape[0]/batch_size)+1), position=0, leave=True):
        # Reset gradients
        optimizer.zero_grad()

        section_ids,center_ids, context_ids, neg_ids,num_contexts,neg_doc_ids = generator.next()
        
        center_ids_tens = torch.LongTensor(center_ids).to(device)
        context_ids_tens = torch.LongTensor(context_ids).to(device)
        section_ids_tens = torch.LongTensor(section_ids).to(device)
        neg_ids_tens = torch.LongTensor(neg_ids).to(device)
        neg_doc_ids_tens = torch.LongTensor(neg_doc_ids).to(device)
        loss = model(center_ids_tens, section_ids_tens, context_ids_tens, neg_ids_tens,neg_doc_ids_tens,num_contexts)
        loss.backward()  # backpropagate loss
        epoc_loss += loss.item()
        optimizer.step()
        
    loss_array.append(epoc_loss*num_batches/vocab.size())
    if loss_array[-1]<=min(loss_array):
        print("Saving...")
        torch.save(model.state_dict(), "D:\\Latent Meaning Cells\\checkpoint_newbatcher.pth")
        
    sleep(0.1)
    print('Epoch={}. Loss={}.'.format(epoch, loss_array[-1]))


# In[100]:


vocab.get_id("women")


# In[20]:


document_embeddings = np.zeros((sub_tokens.shape[0],100))
for i in range(sub_tokens.shape[0]):
    document_embeddings[i] = model.encoder.section_embeddings(torch.Tensor(np.asarray([i])).long().to("cuda")).data.to("cpu").numpy()


# In[79]:


doc = document_embeddings[0]
doc = doc/np.sqrt(np.sum(np.dot(doc,doc)))
normalized = np.divide(document_embeddings,np.sqrt(np.sum(np.multiply(document_embeddings,document_embeddings),axis = 1)).reshape(-1,1))
dist = np.matmul(normalized,doc)


# In[80]:


np.argsort(-dist)[:50]


# ### Similar articles: Months-April

# In[81]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[0]])) 


# In[82]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[337]])) 


# In[83]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[193]])) 


# ### Similar articles: Footbal

# In[62]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[9999]])) 


# In[68]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[81252]])) 


# In[65]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[34080]])) 


# ### Similar articles: Weapons

# In[53]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[5192]])) 


# In[74]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[46029]]))


# In[77]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[2141]]))


# ### Similar articles: Cities/towns - Germany

# In[84]:


doc = document_embeddings[7889]
doc = doc/np.sqrt(np.sum(np.dot(doc,doc)))
normalized = np.divide(document_embeddings,np.sqrt(np.sum(np.multiply(document_embeddings,document_embeddings),axis = 1)).reshape(-1,1))
dist = np.matmul(normalized,doc)
np.argsort(-dist)[:50]


# In[85]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[7889]]))


# In[88]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[91626]]))


# In[89]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[33432]]))


# In[90]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[9092]]))


# In[103]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[9092]]))


# ### Similar articles: Marxism - Political Movements

# In[104]:


doc = document_embeddings[1241]
doc = doc/np.sqrt(np.sum(np.dot(doc,doc)))
normalized = np.divide(document_embeddings,np.sqrt(np.sum(np.multiply(document_embeddings,document_embeddings),axis = 1)).reshape(-1,1))
dist = np.matmul(normalized,doc)
np.argsort(-dist)[:50]


# In[113]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[1241]]))


# In[106]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[74651]]))


# In[107]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[4594]]))


# In[109]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[3651]]))


# In[114]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[10769]]))


# In[120]:


print(' '.join([vocab.get_token(x) for x in sub_tokens[45906]]))


# **Word embeddings:**

# In[93]:


vocab.get_id("marxism")


# In[17]:


word_embeddings = np.zeros((vocab.size(),100))
for i in range(vocab.size()):
    word_embeddings[i] = model.encoder.token_embeddings(torch.Tensor(np.asarray([i])).long().to("cuda")).data.to("cpu").numpy()


# In[69]:


word =  word_embeddings[1]
dist = word_embeddings - word.reshape(1,-1)
dist = np.multiply(dist,dist)
dist = np.sqrt(np.sum(dist,axis = 1))


# In[ ]:


for x in np.argsort(dist)[:50]:
    print(vocab.get_token(x))


# In[72]:


word = word_embeddings[8874]
word = word/np.sqrt(np.sum(np.dot(word,word)))
normalized = np.divide(word_embeddings,np.sqrt(np.sum(np.multiply(word_embeddings,word_embeddings),axis = 1)).reshape(-1,1))
dist = np.matmul(normalized,word)


# In[73]:


for x in np.argsort(-dist)[:25]:
    print(vocab.get_token(x),dist[x],x)


# **Expected Meaning: Test Zone**

# In[48]:


wiki_data.token_doc_matrix[1].todense()


# In[61]:


sparse.diags(1/token_doc_matrix.sum(axis=1).A.ravel())


# In[24]:


token_doc_matrix = sparse.diags(1/token_doc_matrix.sum(axis=1).A.ravel()).dot(token_doc_matrix)


# In[55]:


token_doc_matrix[3,0]


# In[58]:


token_doc_matrix[(1,0)]


# In[15]:


token_doc_matrix = dok_matrix(token_doc_matrix)


# In[16]:


word_meaning = np.zeros((token_doc_matrix.shape[0],100))
for key in tqdm(token_doc_matrix.keys()):
    word_meaning[key[0]] += token_doc_matrix[key]*model.eval().encoder.forward(torch.Tensor(np.asarray([int(key[0])])).long().to("cuda"),torch.Tensor(np.asarray([int(key[1])])).long().to("cuda"))[0].data.to("cpu").numpy()[0]


# In[147]:


vocab.get_id("anarchism")


# In[154]:


[vocab.get_token(x) for x in sub_tokens[0]]


# In[148]:


np.argsort(token_doc_matrix[10661].todense())


# In[18]:


word = model.eval().encoder.forward(torch.Tensor(np.asarray([1])).long().to("cuda"),torch.Tensor(np.asarray([0])).long().to("cuda"))[0].data.to("cpu").numpy()[0]
word = word/np.sqrt(np.sum(np.dot(word,word)))
normalized = np.divide(word_embeddings,np.sqrt(np.sum(np.multiply(word_embeddings,word_embeddings),axis = 1)).reshape(-1,1))
dist = np.matmul(normalized,word)


# In[121]:


vocab.get_id("team")


# In[118]:


from scipy.spatial import distance
april = model.eval().encoder.forward(torch.Tensor(np.asarray([19981])).long().to("cuda"),torch.Tensor(np.asarray([9999])).long().to("cuda"))[0].data.to("cpu").numpy()[0]
res = []
words = []
for key in tqdm(token_doc_matrix.keys(), position=0, leave=True):
    month = model.eval().encoder.forward(torch.Tensor(np.asarray([int(key[0])])).long().to("cuda"),torch.Tensor(np.asarray([int(key[1])])).long().to("cuda"))[0].data.to("cpu").numpy()[0]
    1 - distance.cosine(april, month)
    m = april-month
    if key[0]!=19981 and key[1]!=9999:
        res.append(1/2*np.dot(m,m))
    try:
        if res[-1]<=min(res):
            print("Current word:",vocab.get_token(key[0]),res[-1] , key)
            words.append(key[0])
    except:
        pass


# In[124]:


from scipy.spatial import distance
april = model.eval().encoder.forward(torch.Tensor(np.asarray([19981])).long().to("cuda"),torch.Tensor(np.asarray([9999])).long().to("cuda"))[0].data.to("cpu").numpy()[0]
month = model.eval().encoder.forward(torch.Tensor(np.asarray([2817])).long().to("cuda"),torch.Tensor(np.asarray([9999])).long().to("cuda"))[0].data.to("cpu").numpy()[0]
1 - distance.cosine(april, month)
m = april-month
1/2*np.dot(m,m)


# In[123]:


vocab.get_id("football")


# In[74]:


marxism = model.encoder.token_embeddings(torch.Tensor(np.asarray([8874])).long().to("cuda")).data.to("cpu").numpy()
socialism = model.encoder.token_embeddings(torch.Tensor(np.asarray([15103])).long().to("cuda")).data.to("cpu").numpy()
1-distance.cosine(marxism, socialism)


# In[ ]:


from scipy.spatial import distance
april = model.eval().encoder.forward(torch.Tensor(np.asarray([1])).long().to("cuda"),torch.Tensor(np.asarray([0])).long().to("cuda"))[0].data.to("cpu").numpy()[0]
month = model.eval().encoder.forward(torch.Tensor(np.asarray([2])).long().to("cuda"),torch.Tensor(np.asarray([0])).long().to("cuda"))[0].data.to("cpu").numpy()[0]
distance.cosine(april, month)


# In[19]:


for x in np.argsort(-dist)[:50]:
    print(vocab.get_token(x))


# In[ ]:




