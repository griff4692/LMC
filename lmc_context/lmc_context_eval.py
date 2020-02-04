import re
import sys

import argparse
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/lmc_context/')
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/utils/')
from compute_utils import compute_kl, mask_2D
from lmc_context_utils import restore_model
from model_utils import tensor_to_np

SWORDS = set(stopwords.words('english'))


def create_section_token(section):
    section = re.sub('[:\s]+', '', section).upper()
    return 'header={}'.format(section)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval script for Bayesian LMC Skip-Gram Model')

    # Functional Arguments
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')
    args = parser.parse_args()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    prev_args, model, token_vocab, metadata_vocab, optimizer_state, token_section_counts = restore_model(args.experiment)
    model.eval()
    model.to(args.device)

    # center_word = ['parking']
    # headers = ['header=SOCIALHISTORY', 'header=HPI', 'header=FAMILYHISTORY', 'header=CAMPUS',]
    # compare_words = ['car', 'garage', 'lot', 'field', 'cancer', 'history']

    # center_word = ['cavity']
    # headers = ['<pad>', 'header=ECHOCARDIOGRAM', 'header=ECG', 'header=ECHO', 'header=FAMILYHISTORY',
    #            'header=HISTORYOFPRESENTILLNESS', 'header=CARDIOVASCULAR', 'header=LEFTATRIUM', 'header=CHIEFCOMPLAINT',
    #            'header=LEFTVENTRICLE', 'header=SOCIALHISTORY'
    #            ]
    # compare_words = ['doppler', 'history', 'aorta', 'atrium', 'chamber', 'ventricular', 'lv', 'echocardiogram',
    #                  'cardiogram']

    # center_word = ['atrium']
    # headers = ['<pad>', 'header=LEFTVENTRICLE', 'header=CHIEFCOMPLAINT', 'header=SOCIALHISTORY',
    #            'header=FAMILYHISTORY', 'header=ECHO', 'header=LEFTATRIUM', 'header=RIGHTVENTRICLE',
    #            'header=HISTORYOFPRESENTILLNESS', 'header=IMPRESSION', 'header=DISCHARGEMEDICATIONS']
    # compare_words = ['ventricular', 'history', 'skin', 'cavity', 'chest', 'atria', 'advil', 'mg']

    # center_word = ['pain']
    # headers = ['header=HISTORYPRESENTILLNESS', 'header=DISCHARGEDATE', 'header=CHIEFCOMPLAINT', 'header=DISCHARGEMEDICATIONS', 'header=IMPRESSION']
    # compare_words = ['mg', 'heart', 'smoking', 'meds']

    center_word =['history']
    context_words = ['<pad>']
    headers = ['header=SOCIALHISTORY', 'header=PASTMEDICALHISTORY', 'header=FAMILYHISTORY', 'header=GENERALAPPEARANCE',
               'header=CHIEFCOMPLAINT', 'header=NUTRITION', 'header=LEFTVENTRICLE', 'header=DISPOSITION',
               'header=GLYCEMICCONTROL']
    compare_words = ['smoking', 'diabetes', 'heart', 'depression', 'cholesterol']

    center_word = ['mg']
    headers = [
        'header=DISCHARGEMEDICATIONS', 'header=IMAGING', 'header=REVIEWOFSYSTEMS'
    ]
    compare_words = ['magnesium', 'milligrams', 'myasthenia'] # gravis

    # top_word_idxs = np.where(np.array(token_vocab.support) > 1000)[0]
    # top_words = list(set(list(np.array(token_vocab.i2w)[top_word_idxs])) - SWORDS)
    # compare_words = top_words
    #
    # center_word = ['history']
    # headers = ['<pad>', 'header=SKIN', 'header=CHIEFCOMPLAINT', 'header=SOCIALHISTORY',
    #            'header=FAMILYHISTORY', 'header=ECHO', 'header=LEFTATRIUM', 'header=RIGHTVENTRICLE',
    #            'header=HISTORYOFPRESENTILLNESS', 'header=IMPRESSION', 'header=DISCHARGEMEDICATIONS']
    # compare_words = ['illness', 'pain', 'smoking']

    # center_word = ['history']
    # headers = ['<pad>', 'header=SOCIALHISTORY', 'header=HPI', 'header=PASTMEDICALHISTORY', 'header=FAMILYHISTORY',
    #            'header=CHIEFCOMPLAINT', 'header=HISTORYOFPRESENTILLNESS', 'header=LEFTVENTRICLE']
    # compare_words = ['pain', 'smoking', 'cancer', 'heart', 'past']

    # center_word = ['history']
    # headers = metadata_vocab.i2w[1:]
    # compare_words = ['pain', 'smoking', 'cancer', 'heart']

    center_word_tens = torch.LongTensor([token_vocab.get_id(center_word[0])]).to(device_str)
    header_ids = list(map(lambda x: metadata_vocab.get_id(x), headers))
    compare_ids = list(map(lambda x: token_vocab.get_id(x), compare_words))
    pad_context = torch.zeros([1,]).long().to(device_str)
    context_ids = list(map(lambda x: token_vocab.get_id(x), context_words))

    compare_tens = torch.LongTensor(compare_ids).to(device_str)
    context_tens = torch.LongTensor(context_ids).unsqueeze(0).to(device_str)
    mu_compare, sigma_compare = model.decoder(compare_tens, pad_context.repeat(len(compare_words)))

    mask = mask_2D(torch.Size([1, len(context_ids)]), [len(context_ids)]).to(device_str)

    print('Interpolation between word={} and headers={}'.format(center_word, ', '.join(headers)))
    for i, header_id in enumerate(header_ids):
        header_tens = torch.LongTensor([header_id]).to(device_str)
        print(headers[i])
        for p in np.arange(0, 1.25, 0.25):
            rw = [p, 1.0 - p]
            rel_weights = torch.FloatTensor([
                rw
            ]).to(device_str)

            with torch.no_grad():
                mu_q, sigma_q, weights = model.encoder(center_word_tens, header_tens, context_tens, mask,
                                                       center_mask_p=None, context_mask_p=None, metadata_mask_p=None, rel_weights=rel_weights)

            scores = tensor_to_np(nn.Softmax(-1)(-compute_kl(mu_q, sigma_q, mu_compare, sigma_compare).squeeze(1)))
            order = np.argsort(-scores)
            weight_str = 'Relative Weights --> Word={}.  Section={}'.format(rw[1], rw[0])
            print('\t{}'.format(weight_str))
            for i in order[:min(10, len(order))]:
                print('\t\t{} --> {}'.format(compare_words[i], scores[i]))

    # for i, header_id in enumerate(header_ids):
    #     header_tens = torch.LongTensor([header_id])
    #     with torch.no_grad():
    #         mu_q, sigma_q, weights = model.encoder(center_word_tens, header_tens, context_tens, mask,
    #                                       center_mask_p=None, context_mask_p=None, metadata_mask_p=None)
    #         # m2, s2, w2 = model.encoder(center_word_tens, pad_context, context_tens, mask,
    #         #                               center_mask_p=None, context_mask_p=None, metadata_mask_p=None)
    #
    #         # diff = torch.abs(mu_q - m2)
    #         # print(diff.mean().item(), diff.max().item())
    #         print(weights)
    #     kl = tensor_to_np(compute_kl(mu_q, sigma_q, mu_compare, sigma_compare).squeeze(1))
    #     order = np.argsort(kl)
    #     print(headers[i])
    #     for i in order[:min(10, len(order))]:
    #         print('\t{} --> {}'.format(compare_words[i], kl[i]))

    section_df = pd.read_csv('../preprocess/data/mimic/section.csv').dropna()
    section_names = list(sorted(set(list(section_df.nlargest(100, columns='count')['section'].tolist()))))
    section_keys = list(map(create_section_token, section_names))
    section_ids = metadata_vocab.get_ids(section_keys)
    metadata_embeddings = model.encoder.metadata_embeddings.weight[section_ids, :]

    str = []
    for i, section_name in enumerate(section_names):
        mu = metadata_embeddings[i, :]
        sim = torch.nn.CosineSimilarity()(metadata_embeddings, mu.unsqueeze(0))
        top_section_ids = sim.topk(6).indices.cpu().numpy()
        top_sections = [section_names[i] for i in top_section_ids[1:]]
        print('Closest sections {} --> {}'.format(section_name, ', '.join(top_sections)))
