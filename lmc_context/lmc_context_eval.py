import sys

import argparse
import numpy as np
import torch

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/lmc_context/')
from compute_utils import compute_kl, mask_2D
from lmc_context_utils import restore_model
from model_utils import tensor_to_np


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

    center_word = ['tumor']
    headers = ['header=SOCIALHISTORY', 'header=HPI', 'header=PASTMEDICALHISTORY', 'header=HODGKINSLYMPHOMA',
               'header=CHIEFCOMPLAINT', 'header=HISTORYOFPRESENTILLNESS']
    compare_words = ['melanoma', 'glioblastoma', 'lesion', 'lymphoma', 'nsclc', 'colectomy', 'adenocarcinoma',
                     'hodgkins', 'family']
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

    center_word_tens = torch.LongTensor([token_vocab.get_id(center_word[0])])
    header_ids = list(map(lambda x: metadata_vocab.get_id(x), headers))
    compare_ids = list(map(lambda x: token_vocab.get_id(x), compare_words))
    pad_context = torch.zeros([1,]).long()

    compare_tens = torch.LongTensor(compare_ids)
    mu_compare, sigma_compare = model.decoder(compare_tens, pad_context.repeat(len(compare_words)))

    mask = mask_2D(torch.Size([1, 1]), [1])
    for i, header_id in enumerate(header_ids):
        header_tens = torch.LongTensor([header_id])
        with torch.no_grad():
            mu_q, sigma_q, weights = model.encoder(center_word_tens, header_tens, pad_context.unsqueeze(0), mask,
                                          center_mask_p=None, context_mask_p=None)
            m2, s2, w2 = model.encoder(center_word_tens, pad_context, pad_context.unsqueeze(0), mask,
                                          center_mask_p=None, context_mask_p=None)

            diff = torch.abs(mu_q - m2)
            print(diff.mean().item(), diff.max().item())
            print(weights, w2)
        kl = tensor_to_np(compute_kl(mu_q, sigma_q, mu_compare, sigma_compare).squeeze(1))
        order = np.argsort(kl)
        print(headers[i])
        for i in order:
            print('\t{} --> {}'.format(compare_words[i], kl[i]))

    metadata_embeddings = model.decoder.metadata_embeddings.weight
    metadata = ['header=HISTORYOFPRESENTILLNESS', 'header=DISCHARGEMEDICATIONS', 'header=CHIEFCOMPLAINT',
                'header=ECHO', 'header=LEFTVENTRICLE',
                'header=HPI', 'header=GENERAL', 'header=FAMILYHISTORY', 'header=IMPRESSION']
    str = []
    for section in metadata:
        sid = metadata_vocab.get_id(section)
        mu = metadata_embeddings[sid, :]
        sim = torch.nn.CosineSimilarity()(metadata_embeddings, mu.unsqueeze(0))
        top_section_ids = sim.topk(5).indices.cpu().numpy()
        top_sections = [metadata_vocab.get_token(id) for id in top_section_ids[1:]]
        print('Closest sections {} --> {}'.format(section, ', '.join(top_sections)))
