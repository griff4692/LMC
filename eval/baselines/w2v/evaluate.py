import sys

import argparse
from gensim.models import Word2Vec

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/eval/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
from word_similarity import evaluate_word_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('--eval_fp', default='../../eval_data/')
    parser.add_argument('--experiment', default='debug', help='Save path in weights/ for experiment.')

    args = parser.parse_args()

    negative_samples = [1, 5, 20]
    for ns in negative_samples:
        model = Word2Vec.load('weights/word2vec_ns_{}.model'.format(ns))
        print('\nEvaluations for negative samples={}...'.format(ns))
        word_sim_results = evaluate_word_similarity(model, None, data_dir='../../eval_data', gensim_w2v=True)
        print(model.most_similar(positive='advil'))
