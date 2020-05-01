from collections import Counter, defaultdict
import json
import os
import re
from string import punctuation
import string
import sys
from time import sleep

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

home_dir = os.path.expanduser('~/LMC/')
sys.path.insert(0, os.path.join(home_dir, 'preprocess'))
sys.path.insert(0, os.path.join(home_dir, 'utils'))
from acronym_batcher import AcronymBatcherLoader
from casi_constants import LF_BLACKLIST, LF_MAPPING, SF_BLACKLIST
from mimic_tokenize import clean_text, create_document_token, create_section_token, get_mimic_stopwords, tokenize_str
from model_utils import tensor_to_np


TOKEN_BLACKLIST = set(string.punctuation).union(get_mimic_stopwords()).union(set(['digitparsed']))
# UMLS concept strings include these terms quite frequently
UMLS_BLACKLIST = TOKEN_BLACKLIST.union(set(['unidentified', 'otherwise', 'specified', 'nos', 'procedure']))


def add_section_headers_to_casi():
    """
    :return: None

    CASI section headers to not map 1-1 with MIMIC-III section headers on which the LMC language model is trained.

    To adjust for this, we have curated a manual mapping from CASI section header to MIMIC in the event there is no
    exact match.  This function just adds a column 'section_mapped' to the already preprocessed CASI dataset.
    """
    in_fp = os.path.join(home_dir, 'acronyms/data/casi/preprocessed_dataset_window_10.csv')
    df = pd.read_csv(in_fp)

    freq_sections = pd.read_csv(os.path.join(home_dir, 'preprocess/data/mimic/section_freq.csv'))['section'].tolist()

    section_map = pd.read_csv(os.path.join(home_dir, 'shared_data/casi/casi_mimic_section_map.csv'))
    section_map = section_map.set_index('casi_section')['mimic_section'].to_dict()

    headers = []
    for casi_section in df['section'].tolist():
        if type(casi_section) == float or len(casi_section) == 0:
            headers.append('<pad>')
            continue
        casi_stripped = re.sub(r'\s+', ' ', casi_section).strip(punctuation).strip().upper()

        if casi_stripped in freq_sections:
            new_section = create_section_token(casi_stripped)
        elif casi_stripped in section_map:
            mapped = section_map[casi_stripped]
            if type(mapped) == float or len(mapped) == 0:
                new_section = '<pad>'
            else:
                new_section = create_section_token(mapped)
        else:
            raise Exception('Unaccounted for section!')

        headers.append(new_section)
    df['section_mapped'] = headers
    df.to_csv(in_fp, index=False)


def eval_tokenize(str, unique_only=False):
    """
    :param str: string representing evaluation data
    :param unique_only: Boolean indicating whether to remove duplicate tokens (treat as BoW essentially)
    This should be true for any models which do not rely on seqeuence ordering
    :return: list of tokens
    """
    str = re.sub(r'_%#(\S+)#%_', r'\1', str)
    str = clean_text(str)
    tokens = tokenize_str(str)
    tokens = list(filter(lambda t: t not in TOKEN_BLACKLIST, tokens))

    if unique_only:
        tokens = list(set(tokens))
    return tokens


def load_casi(prev_args, train_frac=1.0):
    """
    :param prev_args: argparse instance from pre-trained language model
    :param train_frac: If you want to fine tune the model, this should be about 0.8.
    :return: train_batcher, test_batcher, train_df, test_df, sf_lf_map

    The sf_lf_map is a dictionary used to get list of candidate LFs (value) for given SF (key)
    """
    casi_dir = os.path.join(home_dir, 'shared_data', 'casi')
    data_fp = os.path.join(casi_dir, 'preprocessed_dataset_window_{}.csv'.format(prev_args.window))
    if not os.path.exists(data_fp):
        print('Need to preprocess dataset first...')
        preprocess_casi_dataset(window=prev_args.window)
        print('Saving dataset to {}'.format(data_fp))
    df = pd.read_csv(data_fp)
    df['section'] = df['section_mapped']
    df['row_idx'] = list(range(df.shape[0]))

    with open(os.path.join(casi_dir, 'sf_lf_map.json'), 'r') as fd:
        sf_lf_map = json.load(fd)

    mimic_fn = os.path.join(home_dir, 'preprocess/context_extraction/data/mimic_rs_dataset_preprocessed_window_10.csv')
    mimic_df = pd.read_csv(mimic_fn)
    mimics_sfs = mimic_df['sf'].unique().tolist()
    used_sf_lf_map = {}
    for sf in mimics_sfs:
        used_sf_lf_map[sf] = sf_lf_map[sf]
    df = df[df['sf'].isin(mimics_sfs)]

    if train_frac == 1.0 or train_frac == 0.0:
        train_batcher = AcronymBatcherLoader(df, batch_size=32)
        test_batcher = AcronymBatcherLoader(df, batch_size=512)
        train_df = df
        test_df = df
    else:
        train_df, test_df = train_test_split(df, test_size=1.0 - train_frac)
        train_batcher = AcronymBatcherLoader(train_df, batch_size=32)
        test_batcher = AcronymBatcherLoader(test_df, batch_size=512)
    return train_batcher, test_batcher, train_df, test_df, used_sf_lf_map


def load_mimic(prev_args, train_frac=1.0):
    """
    :param prev_args: argparse instance from pre-trained language model
    :param train_frac: If you want to fine tune the model, this should be about 0.8.
    :return: train_batcher, test_batcher, train_df, test_df, sf_lf_map

    The sf_lf_map is a dictionary used to get list of candidate LFs (value) for given SF (key)
    """
    casi_dir = os.path.join(home_dir, 'shared_data', 'casi')
    with open(os.path.join(casi_dir, 'sf_lf_map.json'), 'r') as fd:
        sf_lf_map = json.load(fd)
    used_sf_lf_map = {}
    df = pd.read_csv(os.path.join(
        home_dir, 'preprocess/context_extraction/data/mimic_rs_dataset_preprocessed_window_10.csv'))
    df['category'] = df['category'].apply(create_document_token)
    if prev_args.metadata == 'category':
        df['metadata'] = df['category']
    else:
        df['metadata'] = df['section']
        df['metadata'].fillna('<pad>', inplace=True)
    sfs = df['sf'].unique().tolist()
    for sf in sfs:
        used_sf_lf_map[sf] = sf_lf_map[sf]

    if train_frac == 1.0 or train_frac == 0.0:
        train_df, test_df = df, df
        train_batcher = AcronymBatcherLoader(df, batch_size=32)
        test_batcher = AcronymBatcherLoader(df, batch_size=512)
    else:
        train_df, test_df = train_test_split(df, test_size=1.0 - train_frac)
        train_batcher = AcronymBatcherLoader(train_df, batch_size=32)
        test_batcher = AcronymBatcherLoader(test_df, batch_size=512)
    return train_batcher, test_batcher, train_df, test_df, used_sf_lf_map


def lf_tokenizer(str, vocab=None, max_lf_len=5):
    """
    :param str: ';' delimited list of phrases representing a single LF concept (e.g. atrio-ventricular;atrioventricular)
    :param vocab: token vocab from which to look up ids for tokens
    :param max_lf_len: max ngram token length of final LF phrase
    :return: list of de-duped LF tokens

    i.e. 'APOPLEXY;Apoplexy;Apoplexy, NOS;apoplexy;APOPLEXY, CEREBRAL' --> ['apoplexy', 'cerebral']
    """
    tokens_sep = str.split(';')
    token_bag = set()
    token_counts = defaultdict(int)
    for token_str in tokens_sep:
        tokens = tokenize_str(token_str)
        for t in tokens:
            token_bag.add(t)
    token_counts[t] += 1
    token_bag = list(token_bag)
    if vocab is None:
        tokens = list(filter(lambda x: x not in UMLS_BLACKLIST, token_bag))
    else:
        tokens = list(filter(lambda x: x not in UMLS_BLACKLIST and vocab.get_id(x) > -1, token_bag))
    if len(tokens) == 0:
        assert len(token_bag) > 0
        tokens = token_bag
    # If above the max lenght, take most frequent unigrams
    if len(tokens) > max_lf_len:
        available_token_counts = {}
        for t in tokens:
            available_token_counts[t] = token_counts[t]
        truncated_token_counts = Counter(available_token_counts).most_common(max_lf_len)
        tokens = list(map(lambda x: x[0], truncated_token_counts))
    return tokens


def parse_sense_df(sense_fp):
    """
    :param sense_fp: path to the provided CASI sense inventory
    :return: list of LFs as well as mappings between LFs <--> SFs for efficient lookups
    """
    sense_df = pd.read_csv(sense_fp, sep='|')
    sense_df.dropna(subset=['LF', 'SF'], inplace=True)
    lfs = sense_df['LF'].unique().tolist()
    lf_sf_map = {}
    sf_lf_map = defaultdict(set)
    for row_idx, row in sense_df.iterrows():
        row = row.to_dict()
        lf_sf_map[row['LF']] = row['SF']
        sf_lf_map[row['SF']].add(row['LF'])
    for k in sf_lf_map:
        sf_lf_map[k] = list(sorted(sf_lf_map[k]))
    return lfs, lf_sf_map, sf_lf_map


def preprocess_casi_dataset(window=10):
    """
    :param window: target window surrounding SF for which to extract context tokens
    :return: None

    Saves preprocessed dataset to shared_weights directory.
    This function filters, tokenizes, and generates SF -> candidate LF mappings from raw CASI dataset file.
    """
    casi_dir = os.path.join(home_dir, 'shared_data', 'casi')
    in_fp = os.path.join(casi_dir, 'AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt')
    out_fp = os.path.join(casi_dir, 'preprocessed_dataset_window_{}.csv'.format(window))
    df = pd.read_csv(in_fp, sep='|')
    df.dropna(subset=['sf', 'target_lf', 'context'], inplace=True)
    N = df.shape[0]
    print('Dropping SF={}'.format(SF_BLACKLIST))
    df = df[~df['sf'].isin(SF_BLACKLIST)]
    print('Removed {} rows'.format(N - df.shape[0]))
    N = df.shape[0]

    df['lf_in_sf'] = df['sf'].combine(df['target_lf'], lambda sf, lf: sf.lower() in lf.lower().split())
    df = df[~df['lf_in_sf']]
    print('Removed {} rows because the LF is contained within the SF'.format(N - df.shape[0]))
    N = df.shape[0]

    lfs, lf_sf_map, sf_lf_map = parse_sense_df(os.path.join(casi_dir, 'sense_inventory_ii'))
    df['target_lf_sense'] = df['sf'].combine(df['target_lf'], lambda sf, lf: target_lf_sense(lf, sf, sf_lf_map))
    df = df[~df['target_lf_sense'].isin(LF_BLACKLIST) & ~df['target_lf_sense'].isnull()]
    print('Removing {} tokens because LF is in LF_BLACKLIST'.format(N - df.shape[0]))
    df['target_lf_sense'] = df['target_lf_sense'].apply(lambda x: LF_MAPPING[x] if x in LF_MAPPING else x)

    # Tokenize
    sf_occurrences = []  # When multiple of SF in context, keep track of which one the label is for
    tokenized_contexts = []

    valid_rows = []
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        sf_idxs = [m.start() for m in re.finditer(r'\b({})\b'.format(row['sf']), row['context'])]
        target_start_idx = int(row['start_idx'])
        valid = ' ' in row['context'][target_start_idx + len(row['sf']): target_start_idx + len(row['sf']) + 2]
        sf_occurrence_ct = np.where(np.array(sf_idxs) == target_start_idx)[0]
        if not valid or len(sf_occurrence_ct) == 0:
            # print('SF not present in context for row={}'.format(row_idx))
            valid_rows.append(False)
            tokenized_contexts.append(None)
            sf_occurrences.append(None)
        else:
            assert len(sf_occurrence_ct) == 1
            valid_rows.append(True)
            sf_occurrences.append(sf_occurrence_ct[0])
            tokens = eval_tokenize(row['context'])
            tokenized_contexts.append(' '.join(tokens))

    df['valid'] = valid_rows
    df['tokenized_context'] = tokenized_contexts
    df['sf_occurrences'] = sf_occurrences
    prev_n = df.shape[0]
    df = df[df['valid']]
    n = df.shape[0]
    print('Dropping {} rows because we couldn\'t find SF in the context.'.format(prev_n - n))

    trimmed_contexts = []
    print('Tokenizing and extracting context windows...')
    valid = []
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        sf_label_order = int(row['sf_occurrences'])
        tokens = row['tokenized_context'].split()
        sf_idxs = np.where(np.array(tokens) == row['sf'].lower())[0]

        if len(sf_idxs) == 0 or sf_label_order >= len(sf_idxs):
            valid.append(False)
            trimmed_contexts.append(None)
        else:
            valid.append(True)
            sf_idx = sf_idxs[sf_label_order]
            start_idx = max(0, sf_idx - window)
            end_idx = min(sf_idx + window + 1, len(tokens))
            tc = tokens[start_idx:end_idx]
            trimmed_contexts.append(' '.join(tc))

    df['trimmed_tokens'] = trimmed_contexts
    df['valid'] = valid
    prev_n = df.shape[0]
    df = df[df['valid']]
    df.drop(columns='valid', inplace=True)
    print('Issues parsing into tokens for {} out of {} remaining examples'.format(prev_n - df.shape[0], prev_n))
    N = df.shape[0]

    # Remove dominant SFs after parsing
    df.drop_duplicates(subset=['target_lf', 'tokenized_context'], inplace=True)
    print('Removed {} examples with duplicate context-target LF pairs'.format(N - df.shape[0]))
    N = df.shape[0]

    dominant_sfs = set()
    sfs = df['sf'].unique().tolist()
    used_sf_lf_map = {}
    for sf in sfs:
        unique_senses = list(sorted(df[df['sf'] == sf]['target_lf_sense'].unique().tolist()))
        if len(unique_senses) == 1:
            dominant_sfs.add(sf)
        else:
            used_sf_lf_map[sf] = unique_senses
    df = df[~df['sf'].isin(dominant_sfs)]
    print('Removing {} SFs ({} examples) because they have a dominant sense'.format(len(dominant_sfs), N - df.shape[0]))

    df['target_lf_idx'] = df['sf'].combine(
        df['target_lf_sense'], lambda sf, lf_sense: used_sf_lf_map[sf].index(lf_sense))

    print('Finished preprocessing dataset of size={}.  Now saving it to {}'.format(df.shape[0], out_fp))
    df['row_idx'] = list(range(df.shape[0]))
    df.to_csv(out_fp, index=False)

    with open(os.path.join(casi_dir, 'sf_lf_map.json'), 'w') as fd:
        json.dump(used_sf_lf_map, fd)


def process_batch(args, batcher, model, loss_func, token_vocab, metadata_vocab, sf_lf_map, sf_tokenized_lf_map,
                  token_metadata_counts):
    """
    :param args: argparse instance
    :param batcher: AcronymBatcherLoader instance
    :param model: PyTorch acronym expander model from ./modules/
    :param loss_func: PyTorch nn.CrossEntropyLoss function
    :param token_vocab: unigram token vocabulary for MIMIC-III
    :param metadata_vocab: metadata-specific vocabulary for MIMIC-III
    :param sf_lf_map: dictionary mapping SFs to original string LFs
    :param sf_tokenized_lf_map: dictionary mapping SFs to tokenized LFs
    :param token_metadata_counts: dictionary mapping LFs to metadata counts.  Used for computing p(metadata|LF)
    :return: average loss in mini-batch along with other performance metrics

    rel_weight only applies to the LMC model which returns the result of the metadata-token gating function
    """
    batch_input, batch_p, batch_counts = batcher.next(token_vocab, sf_lf_map, sf_tokenized_lf_map,
                                                  token_metadata_counts, metadata_vocab=metadata_vocab)
    batch_input = list(map(lambda x: torch.LongTensor(x).clamp_min_(0).to(args.device), batch_input))
    batch_p = list(map(lambda x: torch.FloatTensor(x).to(args.device), batch_p))
    full_input = batch_input + batch_counts if args.lm_type == 'bsg' else batch_input + batch_p + batch_counts
    scores, target, rel_weights = model(*full_input)
    num_correct = len(np.where(tensor_to_np(torch.argmax(scores, 1)) == tensor_to_np(target))[0])
    num_examples = len(batch_counts[0])
    batch_loss = loss_func.forward(scores, target)
    return batch_loss, num_examples, num_correct, scores, rel_weights


def run_test_epoch(args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_tokenized_lf_map,
               sf_lf_map, token_metadata_counts):
    """
    :param args: argparse instance
    :param test_batcher: AcronymBatcherLoader instance
    :param model: PyTorch acronym expander model from ./modules/
    :param loss_func: PyTorch nn.CrossEntropyLoss function
    :param token_vocab: unigram token vocabulary for MIMIC-III
    :param metadata_vocab: metadata-specific vocabulary for MIMIC-III
    :param sf_tokenized_lf_map: dictionary mapping SFs to tokenized LFs
    :param sf_lf_map: dictionary mapping SFs to original string LFs
    :param token_metadata_counts: dictionary mapping LFs to metadata counts.  Used for computing p(metadata|LF)
    :return: loss averaged across mini-batches from the test set
    """
    test_batcher.reset(shuffle=False)
    test_epoch_loss, test_examples, test_correct = 0.0, 0, 0
    model.eval()
    for _ in tqdm(range(test_batcher.num_batches())):
        with torch.no_grad():
            batch_loss, num_examples, num_correct, _, _ = process_batch(
                args, test_batcher, model, loss_func, token_vocab, metadata_vocab, sf_lf_map, sf_tokenized_lf_map,
                token_metadata_counts)
        test_correct += num_correct
        test_examples += num_examples
        test_epoch_loss += batch_loss.item()
    sleep(0.1)
    test_loss = test_epoch_loss / float(test_batcher.num_batches())
    test_acc = test_correct / float(test_examples)
    print('Test Loss={}. Accuracy={}'.format(test_loss, test_acc))
    sleep(0.1)
    return test_loss, test_acc


def run_train_epoch(args, train_batcher, model, loss_func, optimizer, token_vocab, metadata_vocab, sf_tokenized_lf_map,
                    sf_lf_map, token_metadata_counts):
    """
    :param args: argparse instance
    :param train_batcher: AcronymBatcherLoader instance
    :param model: PyTorch acronym expander model from ./modules/
    :param loss_func: PyTorch nn.CrossEntropyLoss function
    :param optimizer: PyTorch optimizer (i.e. Adam)
    :param token_vocab: unigram token vocabulary for MIMIC-III
    :param metadata_vocab: metadata-specific vocabulary for MIMIC-III
    :param sf_tokenized_lf_map: dictionary mapping SFs to tokenized LFs
    :param sf_lf_map: dictionary mapping SFs to original string LFs
    :param token_metadata_counts: dictionary mapping LFs to metadata counts.  Used for computing p(metadata|LF)
    :return: loss averaged across mini-batches from the test set

    :return: loss averaged across mini-batches from the train set
    """
    train_batcher.reset(shuffle=True)
    train_epoch_loss, train_examples, train_correct = 0.0, 0, 0
    model.train()
    for _ in tqdm(range(train_batcher.num_batches())):
        optimizer.zero_grad()
        batch_loss, num_examples, num_correct, _, _ = process_batch(
            args, train_batcher, model, loss_func, token_vocab, metadata_vocab, sf_lf_map, sf_tokenized_lf_map,
            token_metadata_counts)
        batch_loss.backward()
        optimizer.step()

        # Update metrics
        train_epoch_loss += batch_loss.item()
        train_examples += num_examples
        train_correct += num_correct

    sleep(0.1)
    train_loss = train_epoch_loss / float(train_batcher.num_batches())
    train_acc = train_correct / float(train_examples)
    print('Train Loss={}. Accuracy={}'.format(train_loss, train_acc))
    sleep(0.1)
    return train_loss


def target_lf_sense(target_lf, sf, sf_lf_map):
    """
    :param target_lf: exact string to locate in sense inventory
    :param sf: short form (i.e. acronym)
    :param sf_lf_map: map of SF to sense inventory of LFs
    :return: target_lf stripped of an SF suffix.

    Returns None if one of CASI unsure or mistake sense labels.
    CASI sometimes provides target LFs of the form: <LF, SF>.  For instance, aspirin, ASA.
    This often indicates that the LF is not a true expansion for the SF but in this instance it is the appropriate
    expansion. For these data points, we don't assign the SF as given in the sense inventory but rather use the labels
    and, in the process, remove the SF from the string representation of the LF.

    "aspirin, ASA" --> LF = "aspirin", SF = "ASA"
    """
    if 'UNSURED SENSE' in target_lf or 'MISTAKE' in target_lf or 'GENERAL ENGLISH' in target_lf or 'NAME' in target_lf:
        return None

    target_lf_arr = re.split(r':([a-zA-Z]+)', target_lf)
    target_lf_arr = list(filter(lambda x: len(x) > 0, target_lf_arr))
    proposed_sf = sf if len(target_lf_arr) == 1 else target_lf_arr[1].upper()
    stripped_target_lf = target_lf_arr[0]

    actual_sf = proposed_sf if proposed_sf in sf_lf_map.keys() else sf
    for full_lf in sf_lf_map[actual_sf]:
        for lf in full_lf.split(';'):
            if lf.lower() == stripped_target_lf.lower():
                return full_lf

    return stripped_target_lf
