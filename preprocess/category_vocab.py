# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:12:49 2020

@author: Mert Ketenci
"""

"""
A vocab module tailored for categories
"""
class category_Vocab:
    
    def store_categories(self,categories):
        flat_categories = list(set([item for sublist in categories for item in sublist]))
        self.count = len(flat_categories)
        self.category_to_id = dict()
        self.id_to_category = dict()
        for i in range(len(flat_categories)):
            self.category_to_id[flat_categories[i]] = i
            self.id_to_category[i] = flat_categories[i]
            
    def get_ids(self,x):
        return self.category_to_id[x]
    
    def get_category(self,x):
        return self.id_to_category[x]
    
    def size(self):
        return self.count