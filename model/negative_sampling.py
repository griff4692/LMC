# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:07:25 2019

@author: Mert Ketenci
"""
import numpy as np


class negative_sample:
    
    def __init__(self,words,n_negative):
        
        self.words = np.asarray(words)
        self.n_words = self.words.shape[0]
        self.p_keep = np.asarray([words.tolist().count(w)**(3/4) for w in self.words])
        self.p_keep =  self.p_keep/np.sum(self.p_keep)
        self.n_negative = n_negative
    
    def sample(self):
        
        selected_indexes = np.random.choice(self.words.shape[0],self.n_negative, p=self.p_keep)
        negative_sample = np.asarray([self.words[i] for i in selected_indexes])
        return negative_sample
        