# from collections import defaultdict
# import os
# from shutil import rmtree
# import sys
# from time import sleep
#
# from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
# from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import (
#     BidirectionalLanguageModelTokenEmbedder)
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
# from tqdm import tqdm
#
# home_dir = os.path.expanduser('~/LMC/')
# os.environ['BIDIRECTIONAL_LM_VOCAB_PATH'] = os.path.join('modules', 'elmo')
# os.environ['BIDIRECTIONAL_LM_TRAIN_PATH'] = os.path.join('modules', 'elmo')
#
# sys.path.insert(0, os.path.join(home_dir, 'acronyms', 'modules'))
# sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
# sys.path.insert(0, os.path.join(home_dir, 'utils'))
# from elmo_acronym_expander import ELMoAcronymExpander
# from error_analysis import elmo_analyze
# from eval_utils import lf_tokenizer
# from evaluate import load_casi, load_mimic
# from model_utils import tensor_to_np
#
#
# def get_pretrained_elmo(lm_model_file='~/allennlp-0.9.0/output_path/model.tar.gz'):
#     return BidirectionalLanguageModelTokenEmbedder(
#         archive_file=lm_model_file,
#         dropout=0.2,
#         bos_eos_tokens=["<S>", "</S>"],
#         remove_bos_eos=True,
#         requires_grad=True
#     )
#
#
# def elmo_evaluate(args, loader):
#     args.metadata = None
#     train_batcher, test_batcher, train_df, test_df, used_sf_lf_map = loader(args)
#
#     # Create model experiments directory or clear if it already exists
#     weights_dir = os.path.join(home_dir, 'weights', 'acronyms', args.experiment)
#     if os.path.exists(weights_dir):
#         print('Clearing out previous weights in {}'.format(weights_dir))
#         rmtree(weights_dir)
#     os.mkdir(weights_dir)
#     results_dir = os.path.join('../acronyms', weights_dir, 'results')
#     os.mkdir(results_dir)
#     os.mkdir(os.path.join(results_dir, 'confusion'))
#
#     elmo = get_pretrained_elmo()
#     device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = ELMoAcronymExpander(elmo).to(device_str)
#     indexer = ELMoTokenCharactersIndexer()
#     vocab = elmo._lm.vocab
#
#     sf_tokenized_lf_map = defaultdict(list)
#     for sf, lf_list in used_sf_lf_map.items():
#         for lf in lf_list:
#             tokens = lf_tokenizer(lf)
#             sf_tokenized_lf_map[sf].append(tokens)
#
#     # Instantiate Adam optimizer
#     trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
#     print('{} trainable params'.format(len(trainable_params)))
#     optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
#     loss_func = nn.CrossEntropyLoss()
#
#     best_weights = None
#     best_epoch = 1
#     lowest_test_loss = run_test_epoch(model, test_batcher, indexer, vocab, sf_tokenized_lf_map, loss_func)
#     elmo_analyze(test_batcher, model, used_sf_lf_map, vocab, sf_tokenized_lf_map, indexer, results_dir=results_dir)
#
#     for epoch in range(1, args.epochs + 1):
#         sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
#         print('Starting Epoch={}'.format(epoch))
#
#         model.train()  # just sets .requires_grad = True
#         train_batcher.reset(shuffle=True)
#         train_epoch_loss, train_examples, train_correct = 0.0, 0, 0
#         for _ in tqdm(range(train_batcher.num_batches())):
#             optimizer.zero_grad()
#
#             batch_input, num_outputs = train_batcher.elmo_next(vocab, indexer, sf_tokenized_lf_map)
#             batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0).to('cuda'), batch_input))
#             scores, target = model(*batch_input + [num_outputs])
#             num_correct = len(np.where(tensor_to_np(torch.argmax(scores, 1)) == tensor_to_np(target))[0])
#             num_examples = len(num_outputs)
#             batch_loss = loss_func.forward(scores, target)
#
#             batch_loss.backward()
#             optimizer.step()
#
#             train_correct += num_correct
#             train_examples += num_examples
#             train_epoch_loss += batch_loss.item()
#
#         sleep(0.1)
#         train_loss = train_epoch_loss / float(train_batcher.num_batches())
#         train_acc = train_correct / float(train_examples)
#         print('Train Loss={}. Accuracy={}'.format(train_loss, train_acc))
#         sleep(0.1)
#
#         test_loss = run_test_epoch(model, test_batcher, indexer, vocab, sf_tokenized_lf_map, loss_func)
#         lowest_test_loss = min(lowest_test_loss, test_loss)
#         if lowest_test_loss == test_loss:
#             best_weights = model.state_dict()
#             best_epoch = epoch
#
#     print('Loading weights from {} epoch to perform error analysis'.format(best_epoch))
#     model.load_state_dict(best_weights)
#     elmo_analyze(test_batcher, model, used_sf_lf_map, vocab, sf_tokenized_lf_map, indexer, results_dir=results_dir)
#
#
# def run_test_epoch(model, test_batcher, indexer, vocab, sf_tokenized_lf_map, loss_func):
#     model.eval()  # just sets .requires_grad = True
#     test_batcher.reset(shuffle=False)
#     test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
#     for _ in tqdm(range(test_batcher.num_batches())):
#         batch_input, num_outputs = test_batcher.elmo_next(vocab, indexer, sf_tokenized_lf_map)
#         batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0).to('cuda'), batch_input))
#         with torch.no_grad():
#             scores, target = model(*batch_input + [num_outputs])
#         num_correct = len(np.where(tensor_to_np(torch.argmax(scores, 1)) == tensor_to_np(target))[0])
#         num_examples = len(num_outputs)
#         batch_loss = loss_func.forward(scores, target)
#
#         test_correct += num_correct
#         test_examples += num_examples
#         test_epoch_loss += batch_loss.item()
#     sleep(0.1)
#     test_loss = test_epoch_loss / float(test_batcher.num_batches())
#     test_acc = test_correct / float(test_examples)
#     print('Test Loss={}. Accuracy={}'.format(test_loss, test_acc))
#     sleep(0.1)
#     return test_loss
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('Main script for Acronym Evaluation on Pre-Trained ELMo')
#
#     # Functional Arguments
#     parser.add_argument('-debug', action='store_true', default=False)
#     parser.add_argument('--experiment', default='elmo', help='Save path in weights/ for experiment.')
#     parser.add_argument('--dataset', default='casi', help='casi or mimic')
#
#     # Training Hyperparameters
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser.add_argument('--epochs', default=0, type=int)
#     parser.add_argument('--lr', default=0.001, type=float)
#     parser.add_argument('--window', default=10, type=int)
#     args = parser.parse_args()
#
#     args.experiment += '_{}'.format(args.dataset)
#     loader = load_casi if args.dataset.lower() == 'casi' else load_mimic
#     elmo_evaluate(args, loader)
