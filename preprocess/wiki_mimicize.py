# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:11:28 2020

@author: Mert Ketenci
"""
import argparse
from gensim.corpora import WikiCorpus
import glob,os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'D:/ClinicalBayesianSkipGram/utils')
from model_utils import render_args
sys.path.insert(0, 'D:/ClinicalBayesianSkipGram/')


class extract_data:

    def __init__(self,data_path,keep_prob):
        self.X = []
        self.vocab = []
        self.tokens = []
        self.keep_prob = []
        self.subsampled_ids = []
        self.tokenized_subsampled_data = []
        self.data_path = data_path
        self.keep_prob = keep_prob
        
    def make_corpus(self):
        print("Loading data")
        wiki = WikiCorpus(self.data_path + "enwiki-latest-pages-articles.xml.bz2")
        print("Processing...")
        i = 1
        for document in wiki.get_texts():
            out_f_ = ''.join([out_f,'wiki_en_',str(i),".txt"])
            output = open(out_f_, 'w',encoding='utf-8')
            output.write(' '.join(document))
            output.close()
            i+=1
            if i%10000 == 0:  
                print(i,"documents procecessed")
        print('Processing complete!')
    
    def read_data(self):
        wikipedia = []
        keep_prob = self.keep_prob
        print("Reading txt data")
        os.chdir(self.data_path)
        N = len(glob.glob("*.txt"))
        #choice = np.random.choice(N, sample_size, replace=False)+1
        for i in tqdm(range(1,N+1), position=0, leave=True):
            if np.random.binomial(1, keep_prob):
                text = ''.join([self.data_path,"wiki_en_",str(i),".txt"])
                with open(text,'r',encoding='utf-8') as f:
                    wikipedia.append(''.join(list(f)).lower())
            else:
                pass
        print("Saving sampled wiki data, sample rate:",keep_prob)
        wikipedia = pd.DataFrame(wikipedia,columns = ["TEXT"])
        wikipedia["CATEGORY"] = wikipedia.index.tolist()
        wikipedia.to_csv('NOTEEVENTS_clean.csv')
        
if __name__ == '__main__':
    arguments = argparse.ArgumentParser('MIMIC (v3) Note Tokenization.')
    arguments.add_argument('--wiki_fp', default='data/simplewiki/')
    arguments.add_argument('-debug', default=False, action='store_true')
    arguments.add_argument('-combine_phrases', default=False, action='store_true')
    arguments.add_argument('-split_sentences', default=False, action='store_true')
        
    args = arguments.parse_args()
    render_args(args)
    
    args.wiki_fp = sys.path[0] + args.wiki_fp
    wiki_data = extract_data(args.wiki_fp,keep_prob = 1)
    
    #wiki_data.make_corpus() #Run this cell only if you are going to use entire wiki
    wiki_data.read_data() #If you have txt files alread running this would be enough
