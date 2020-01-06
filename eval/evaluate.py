import sys

import argparse
import torch

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/acronyms/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from fine_tune import acronyms_finetune
from model_utils import restore_model
from word_similarity import evaluate_word_similarity


def evaluate(args):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    prev_args, vae_model, vocab, optimizer_state = restore_model(args.experiment)

    # Make sure it's NOT calculating gradients
    model = vae_model.to(args.device)
    model.eval()  # just sets .requires_grad = False

    print('\nEvaluations...')
    evaluate_word_similarity(model, vocab, combine_phrases=prev_args.combine_phrases)
    args.bsg_experiment = prev_args.experiment  # Tell which model to pull from
    # TODO integrate these better
    args.epochs = 5
    args.batch_size = 32
    args.debug = prev_args.debug
    args.lr = 0.001
    args.random_priors = False
    args.random_encoder = False
    args.use_att = False
    args.att_style = None
    acronyms_finetune(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')

    args = parser.parse_args()
    evaluate(args)
