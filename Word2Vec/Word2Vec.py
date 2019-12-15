import csv
import sys
import numpy as np
sys.path.insert(0, 'D:\Github\ClinicalBayesianSkipGram\\total\Word2Vec')
import argparse
from glob import glob
import os
from gensim.models import Word2Vec
import pickle
from vocab import Vocab
from scipy.spatial.distance import cosine

def partition_ids(ids,dump_path,mode = 1000000, include_headers = False):
    start = ""
    end = ""
    list_ids = []
    j = 0
    for i in range(ids.shape[0]):
        if ids[i] == 0 and start =="":
            start = i
        elif ids[i] == 0 and end == "":
            end = i
        if start !="" and end !="":
            if include_headers:
                list_ids.append([str(x) for x in ids[start+1:end]])
            else:
                list_ids.append([str(x) for x in ids[start+1:end] if x>0])
            start = end
            end = ""
        if i%mode == 0:
            filename = "".join([dump_path,"list_ids",str(j),".csv"])
            with open(filename, 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(list_ids)
            list_ids = []
            j += 1

def read_ids(ids_infile ="data\ids.npy"):
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)
        if not ids[0] == 0:
            ids = np.insert(ids, [0], 0)
    return ids

def combine_ids(dump_path):
    extension = 'csv'
    os.chdir(dump_path)
    num_files = len(glob('*.{}'.format(extension)))  
    X_train = []
    for j in range(1,num_files):
        filename = "".join([dump_path,"\list_ids",str(j),".csv"])
        f = open(filename, 'r')
        reader = csv.reader(f)
        for row in reader:
            X_train.append(row)
        f.close()
    return X_train

def get_known_ids(vocab, tokens):
    return list(filter(lambda id: id > -1, list(map(lambda tok: vocab.get_id(tok), tokens.split(' ')))))

def point_similarity(model, vocab, tokens_a, tokens_b):
    ids_a = get_known_ids(vocab, tokens_a)
    ids_b = get_known_ids(vocab, tokens_b)

    if len(ids_a) == 0 or len(ids_b) == 0:
        return 0.0

    rep_a = model[str(ids_a)]
    rep_b = model[str(ids_b)]
    sim = 1.0 - cosine(rep_a, rep_b)
    return sim

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Main Script to train Word2Vec')
    parser.add_argument('-partition_data', default=False)
    args = parser.parse_args()
    
    path = sys.path[0]
    dump_path = "".join([path,"\data"])
    ids_infile = "".join([dump_path,"\ids.npy"])
    
    if args.partition_data:
        ids = read_ids(ids_infile)
        
        
        partition_ids(ids, dump_path = dump_path, mode = 1000000, include_headers = False)
    
    print('Loading vocabulary...')
    vocab_infile = "".join([dump_path,"\\vocab.pk"])
    with open(vocab_infile, 'rb') as fd:
        vocab = pickle.load(fd)
    token_vocab_size = vocab.size()
    print('Loaded vocabulary of size={}...'.format(token_vocab_size))
    
    X_train = combine_ids(dump_path)
    model = Word2Vec(X_train, size=100, window=5, min_count=1, workers=4,sample = 1)
    
    print('lesion', 'nodule', point_similarity(model, vocab, 'lesion', 'nodule'))
    print('lesion', 'advil', point_similarity(model, vocab, 'lesion', 'advil'))
    print('pain', 'tolerance', point_similarity(model, vocab, 'pain', 'tolerance'))
    print('advil', 'tylenol', point_similarity(model, vocab, 'advil', 'tylenol'))
    print('advil', 'ibruprofen', point_similarity(model, vocab, 'advil', 'ibruprofen'))
    print('headache', 'migraine', point_similarity(model, vocab, 'headache', 'migraine'))
    print('tolerance', 'lesion', point_similarity(model, vocab, 'tolerance', 'lesion'))

    
            

        
    