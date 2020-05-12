import csv
import os
import pickle
from shutil import rmtree
import sys
from time import sleep, time

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, AdamW

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'acronyms'))
sys.path.insert(0, os.path.join(home_dir, 'acronyms/modules'))
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from acronym_utils import load_mimic, load_casi, load_columbia
from compute_sections import enumerate_metadata_ids_lmc
from evaluate import run_evaluation
from lmc_acronym_expander import LMCAcronymExpander
from lmc_model import LMC, LMCBERT
from lmc_prebatch import create_tokenizer_maps, DistributedDataset, generate_metadata_samples
from lmc_utils import restore_model, save_checkpoint
from model_utils import block_print, enable_print, get_git_revision_hash, render_args, render_num_params
from vocab import Vocab


def _prepare_data(args, token_vocab, ids):
    """
    :param args: argparse instance
    :param token_vocab: vocabulary object storing token unigrams as well as metadata
    :param ids: flattened list of token and metadata ids
    :return: kwargs which hold many useful data structures for efficient access of ids and metadata
    """
    print('Collecting metadata information for {}...'.format(args.metadata))
    assert token_vocab.section_start_vocab_id <= token_vocab.category_start_vocab_id
    start_id = token_vocab.section_start_vocab_id if args.metadata == 'section' else token_vocab.category_start_vocab_id
    end_id = token_vocab.category_start_vocab_id if args.metadata == 'section' else token_vocab.size()
    metadata_id_range = np.arange(start_id, end_id)
    is_metadata = np.isin(ids, metadata_id_range)
    all_metadata_pos_idxs = np.where(ids >= token_vocab.section_start_vocab_id)[0]
    metadata_vocab = Vocab()
    for id in metadata_id_range:
        name = token_vocab.get_token(id)
        metadata_vocab.add_token(name)
    print('Enumerating over ids to get per id metadata info...')
    metadata_pos_idxs = np.where(is_metadata)[0]
    full_metadata_ids, token_metadata_counts = enumerate_metadata_ids_lmc(
        ids, metadata_pos_idxs, token_vocab, metadata_vocab)
    ids[all_metadata_pos_idxs] = -1  # Be very clear that these are special tokens

    print('Generating samples...')
    # It's very expensive to Monte Carlo sample for metadata samples online so we pre-compute a large batch of
    # random samples of size args.metadata_samples and simple iterate through the list when preparing batches
    token_metadata_samples = generate_metadata_samples(
        token_metadata_counts, metadata_vocab, sample=args.metadata_samples)

    # Once we create metadata_vocab, we remove metadata tokens from token_vocab
    token_vocab.truncate(token_vocab.section_start_vocab_id)
    token_vocab_size = token_vocab.size()
    wp_conversions = {}  # Only applicable with BERT
    # Bert Tokenizer if using BERT with flag -bert
    bert_tokenizer_fn = os.path.join(home_dir, 'preprocess/data/bert_tokenizer_vocab.pth')
    bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_fn) if args.bert else None
    if bert_tokenizer is not None:
        metadata_tokens = metadata_vocab.i2w[1:] + ['digitparsed']
        special_tokens_dict = {'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]', 'bos_token': '[BOS]',
                               'eos_token': '[EOS]', 'pad_token': '[PAD]', 'mask_token': '[MASK]',
                               'additional_special_tokens': metadata_tokens}
        num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)
        print('Readded special tokens={}'.format(num_added_toks))
        print('Mapping regular vocab ids to WordPiece ids for token and metadata...')
        # This is just for efficiency so we only compute word pieces once for every unigram in corpus
        wp_conversions = create_tokenizer_maps(bert_tokenizer, token_vocab, metadata_vocab)
        token_vocab_size = max(bert_tokenizer.vocab_size, max(bert_tokenizer.all_special_ids) + 1)
    num_ids = len(ids)
    print('Shuffling data...')
    all_batch_idxs = np.array(list(set(np.arange(num_ids)) - set(all_metadata_pos_idxs)), dtype=int)
    np.random.shuffle(all_batch_idxs)
    num_batches = len(all_batch_idxs) // args.batch_size
    truncated_N = args.batch_size * num_batches
    batches = all_batch_idxs[:truncated_N].reshape(num_batches, args.batch_size)

    # Trigger negative sample categorical parameters to be computed based on corpus support
    token_vocab.neg_sample(size=(1,))
    neg_sample_p = token_vocab.cached_neg_sample_prob

    kwargs = {
        'batches': batches,
        'bert': args.bert,  # Whether or not to use BERT as Encoder & Decoder (ALBERT, more specifically)
        'bert_tokenizer': bert_tokenizer,
        'full_metadata_ids': full_metadata_ids,
        'ids': ids,
        'metadata_vocab': metadata_vocab,
        'neg_sample_p': neg_sample_p,
        'token_metadata_counts': token_metadata_counts,
        'token_metadata_samples': token_metadata_samples,
        'token_vocab': token_vocab,
        'token_vocab_size': token_vocab_size,
        'window_size': args.window,
        'wp_conversions': wp_conversions
    }
    return kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for training Latent Meaning Cells (LMC) model')

    # Functional Arguments
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-debug_model', action='store_true', default=False)
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_gpu', default=1, type=int)

    # Model Hyperparameters
    parser.add_argument('-bert', default=False, action='store_true')
    parser.add_argument('--metadata', default='section',
                        help='sections or category. What to define latent variable over.')
    parser.add_argument('--metadata_samples', default=3, type=int)
    parser.add_argument('--window', default=10, type=int)
    parser.add_argument('-pool_bert', default=False, action='store_true')
    parser.add_argument('-restore', default=False, action='store_true')

    args = parser.parse_args()
    args.git_hash = get_git_revision_hash()
    if args.debug:  # Mini dataset may have fewer than 200 examples
        args.batch_size = 200
    render_args(args)

    # Load Data
    debug_str = '_mini' if args.debug else ''
    bert_str = '_bert' if args.bert else ''
    model_prototype = LMCBERT if args.bert else LMC

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_str)
    print('Training on {}...'.format(device_str))

    # If we are using multiple GPUs, let's keep a uniform batch_size for each GPU
    # Or else, there is no real speed-up gain from using multiple GPUs
    if args.num_gpu > 1 and torch.cuda.device_count() > 1:
        args.batch_size *= torch.cuda.device_count()

    ids_infile = os.path.join(home_dir, 'preprocess', 'data', 'ids{}.npy'.format(debug_str))
    print('Loading data from {}...'.format(ids_infile))
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)

    # Load Vocabulary
    vocab_infile = os.path.join(home_dir, 'preprocess', 'data', 'vocab{}.pk'.format(debug_str))
    print('Loading vocabulary from {}...'.format(vocab_infile))
    with open(vocab_infile, 'rb') as fd:
        token_vocab = pickle.load(fd)
    print('Loaded vocabulary of size={}...'.format(token_vocab.section_start_vocab_id))

    kwargs = _prepare_data(args, token_vocab, ids)
    dataset = DistributedDataset(**kwargs)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Instantiate PyTorch LMC Model
    if args.restore:
        print('Restoring from latest checkpoint...')
        epoch_shift = 5  # TODO determine from checkpoints
        _, model, _, _, _, optimizer_state, _ = restore_model(args.experiment)
    else:
        epoch_shift = 0
        model = model_prototype(  # Either LMC or LMCBERT
            args, kwargs['token_vocab_size'], metadata_vocab_size=kwargs['metadata_vocab'].size())
        optimizer_state = None
    model = model.to(args.device)
    render_num_params(model, kwargs['metadata_vocab'].size())

    num_gpu_available = torch.cuda.device_count()
    if args.num_gpu > 1 and num_gpu_available > 1:
        print("Let's use", num_gpu_available, "GPUs!")
        model = nn.DataParallel(model)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join(home_dir, 'weights', 'lmc', args.experiment)
    # if not args.restore:
    #     if os.path.exists(weights_dir):
    #         print('Clearing out previous weights in {}'.format(weights_dir))
    #         rmtree(weights_dir)
    #     os.mkdir(weights_dir)

    metric_cols = ['examples', 'lm_kl', 'lm_recon', 'epoch', 'hours', 'dataset', 'log_loss', 'accuracy', 'macro_f1',
                   'weighted_f1']
    metrics_file = open(os.path.join(weights_dir, 'metrics.csv'), mode='a')
    metrics_writer = csv.writer(metrics_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    metrics_writer.writerow(metric_cols)
    metrics_file.flush()

    start_time = time()

    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    for epoch in range(1 + epoch_shift, args.epochs + epoch_shift + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        num_batches = len(data_loader)
        epoch_joint_loss, epoch_kl_loss, epoch_recon_loss = 0.0, 0.0, 0.0
        for i, batch_ids in tqdm(enumerate(data_loader), total=num_batches):
            # Reset gradients
            optimizer.zero_grad()
            batch_ids = list(map(lambda x: x[0].to(args.device), batch_ids))
            kl_loss, recon_loss = model(*batch_ids, num_metadata_samples=args.metadata_samples)
            if len(kl_loss.size()) > 0:
                kl_loss = kl_loss.mean(0)
            if len(recon_loss.size()) > 0:
                recon_loss = recon_loss.mean(0)
            joint_loss = kl_loss + recon_loss
            joint_loss.backward()  # backpropagate loss
            clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_joint_loss += joint_loss.item()

            checkpoint_interval = 10000
            if (i + 1) % checkpoint_interval == 0:
                duration_in_hours = (time() - start_time) / (60. * 60.)
                full_example_ct = (((epoch - 1) * float(num_batches)) + i + 1) * args.batch_size
                print('Saving Checkpoint at Batch={}'.format(i + 1))
                d = float(i + 1)
                # Serializing everything from model weights and optimizer state, to to loss function and arguments
                losses_dict = {'losses': {'joint': epoch_joint_loss / d,
                                          'kl': epoch_kl_loss / d,
                                          'recon': epoch_recon_loss / d}
                               }
                print(losses_dict)
                checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
                if epoch < 10:
                    save_checkpoint(args, model, optimizer, token_vocab, losses_dict, kwargs['token_metadata_counts'],
                                    checkpoint_fp=checkpoint_fp, metadata_vocab=kwargs['metadata_vocab'])

                experiments = [(load_casi, 'casi'), (load_mimic, 'mimic'), (load_columbia, 'columbia')]
                prev_epoch_ct = args.epochs
                prev_batch_size = args.batch_size
                for loader, dataset in experiments:
                    args.lm_type = 'lmc'
                    args.lm_experiment = args.experiment
                    args.ckpt = None
                    args.device = device_str
                    args.epochs = 0
                    args.dataset = dataset
                    args.batch_size = 128
                    block_print()
                    metrics = run_evaluation(args, LMCAcronymExpander, loader, restore_model, train_frac=0)
                    enable_print()
                    metrics['dataset'] = dataset
                    metrics['hours'] = duration_in_hours
                    metrics['examples'] = full_example_ct
                    metrics['epoch'] = epoch
                    metrics['lm_recon'] = losses_dict['losses']['recon']
                    metrics['lm_kl'] = losses_dict['losses']['kl']
                    row = [metrics[col] for col in metric_cols]
                    metrics_writer.writerow(row)
                    print(metric_cols)
                    print(row)
                    metrics_file.flush()
                args.batch_size = prev_batch_size
                args.epochs = prev_epoch_ct

        epoch_joint_loss /= float(num_batches)
        epoch_kl_loss /= float(num_batches)
        epoch_recon_loss /= float(num_batches)
        sleep(0.1)
        print('Epoch={}. Joint loss={}.  KL Loss={}. Reconstruction Loss={}'.format(
            epoch, epoch_joint_loss, epoch_kl_loss, epoch_recon_loss))
        sleep(0.1)

        # Serializing everything from model weights and optimizer state, to to loss function and arguments
        losses_dict = {'losses': {'joint': epoch_joint_loss, 'kl': epoch_kl_loss, 'recon': epoch_recon_loss}}
        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        if epoch < 10:  # Epoch >= 10 usually only happens when debugging in which we case we don't want to keep saving
            save_checkpoint(args, model, optimizer, token_vocab, losses_dict, kwargs['token_metadata_counts'],
                            checkpoint_fp=checkpoint_fp, metadata_vocab=kwargs['metadata_vocab'])
    metrics_file.close()
