import sys

import argparse
import torch

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/acronyms/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/bsg/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/lmc_context/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from fine_tune import acronyms_finetune, load_casi, load_mimic
from bsg_utils import restore_model as restore_bsg, save_checkpoint as save_bsg
from lmc_context_utils import restore_model as restore_lmc, save_checkpoint as save_lmc
from acronym_expander import AcronymExpander
from word_similarity import evaluate_word_similarity


def evaluate(args):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    restore_func = restore_bsg if args.lm_type == 'bsg' else restore_lmc
    save_func = save_bsg if args.lm_type == 'bsg' else save_lmc

    if args.lm_type == 'bsg':
        prev_args, lm_model, token_vocab, optimizer_state = restore_func(args.experiment)
    else:
        prev_args, lm_model, token_vocab, _, _, _ = restore_func(args.experiment)

    # Make sure it's NOT calculating gradients
    model = lm_model.to(args.device)
    model.eval()  # just sets .requires_grad = False

    print('\nEvaluations...')
    evaluate_word_similarity(model, token_vocab, combine_phrases=prev_args.combine_phrases)
    args.lm_experiment = prev_args.experiment  # Tell which model to pull from
    # TODO integrate these better
    args.epochs = 5
    args.batch_size = 32
    args.debug = prev_args.debug
    args.lr = 0.001
    args.random_priors = False
    args.random_encoder = False
    args.use_att = False
    args.att_style = None
    args.lm_type = 'bsg'
    print('Fine Tuning on CASI')
    acronyms_finetune(args, AcronymExpander, load_casi, restore_func, save_func)
    print('Fine Tuning on MIMIC Reverse Substitution')
    acronyms_finetune(args, AcronymExpander, load_mimic, restore_func, save_func)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')
    parser.add_argument('--lm_type', default='bsg', help='Save path in weights/ for experiment.')

    args = parser.parse_args()
    evaluate(args)
