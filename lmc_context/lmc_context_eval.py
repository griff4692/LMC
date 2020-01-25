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

    # center_word = ['cavity']
    # headers = ['header=ECHOCARDIOGRAM', 'header=ECG', 'header=ECHO', 'header=FAMILYHISTORY',
    #            'header=HISTORYOFPRESENTILLNESS', 'header=CARDIOVASCULAR', 'header=LEFTATRIUM', 'header=CHIEFCOMPLAINT',
    #            'header=LEFTVENTRICLE', 'header=SOCIALHISTORY'
    #            ]
    # compare_words = ['doppler', 'history', 'aorta', 'atrium', 'chamber', 'ventricular', 'lv', 'echocardiogram',
    #                  'cardiogram']

    # center_word = ['pleural']
    # headers = ['header=LEFTVENTRICLE', 'header=SKIN', 'header=CHIEFCOMPLAINT', 'header=SOCIALHISTORY',
    #            'header=FAMILYHISTORY', 'header=ECHO', 'header=LEFTATRIUM', 'header=RIGHTVENTRICLE',
    #            'header=HISTORYOFPRESENTILLNESS', 'header=IMPRESSION', 'header=DISCHARGEMEDICATIONS']
    # compare_words = ['cancer', 'effusion', 'history', 'skin', 'cavity', 'chest', 'atrium', 'advil']
    #
    # center_word = ['left']
    # headers = ['header=LEFTVENTRICLE', 'header=CHIEFCOMPLAINT', 'header=DISCHARGEMEDICATIONS', 'header=IMPRESSION']
    # compare_words = ['ventricular', 'right', 'mg', 'prilosec', 'pain', 'side']

    # center_word = ['cancer']
    # headers = ['header=SOCIALHISTORY', 'header=HPI', 'header=PASTMEDICALHISTORY', 'header=HODGKINSLYMPHOMA',
    #            'header=CHIEFCOMPLAINT', 'header=HISTORYOFPRESENTILLNESS']
    # compare_words = ['melanoma', 'glioblastoma', 'lymphoma', 'nsclc', 'colectomy', 'adenocarcinoma', 'hodgkins']

    # center_word = ['history']
    # headers = ['header=SOCIALHISTORY', 'header=HPI', 'header=PASTMEDICALHISTORY', 'header=FAMILYHISTORY',
    #            'header=CHIEFCOMPLAINT', 'header=HISTORYOFPRESENTILLNESS', 'header=LEFTVENTRICLE']
    # compare_words = ['pain', 'smoking', 'cancer', 'heart']

    center_word = ['lv']
    headers = metadata_vocab.i2w[1:]
    compare_words = ['ventricle', 'left', 'prothrombin']

    center_word_tens = torch.LongTensor([token_vocab.get_id(center_word[0])])
    header_ids = list(map(lambda x: metadata_vocab.get_id(x), headers))
    compare_ids = list(map(lambda x: token_vocab.get_id(x), compare_words))
    pad_context = torch.zeros([1,]).long()

    compare_tens = torch.LongTensor(compare_ids)
    mu_compare, sigma_compare = model.decoder(
        compare_tens, pad_context.repeat(len(compare_words)))  #, center_mask_p=None, metadata_mask_p=None)

    mask = mask_2D(torch.Size([1, 1]), [1])
    for i, header_id in enumerate(header_ids):
        header_tens = torch.LongTensor([header_id])
        mu_q, sigma_q = model.encoder(center_word_tens, header_tens, pad_context.unsqueeze(0), mask)
                                      # center_mask_p=None, metadata_mask_p=None, context_mask_p=None)
        # mu_q2, sigma_q2 = model.encoder(center_word_tens, pad_context, pad_context.unsqueeze(0), mask,
        #                               center_mask_p=None, metadata_mask_p=None, context_mask_p=None)
        kl = tensor_to_np(compute_kl(mu_q, sigma_q, mu_compare, sigma_compare).squeeze(1))
        order = np.argsort(kl)
        print(headers[i])
        for i in order:
            print('\t{} --> {}'.format(compare_words[i], kl[i]))

    # top_n = min(10000, token_vocab.size())
    # top_word_ids = list(range(top_n))
    # section_ids = np.zeros([top_n,  metadata_vocab.size()])
    # section_p = np.zeros([top_n, metadata_vocab.size()])
    # for i in top_word_ids:
    #     if i > 0:
    #         sids, sp = token_section_counts[i]
    #         section_ids[i, :len(sids)] = sids
    #         section_p[i, :len(sp)] = sp
    #
    # center_ids_tens = torch.LongTensor(top_word_ids).unsqueeze(-1).to(args.device)
    # section_ids_tens = torch.LongTensor(section_ids).to(args.device)
    # section_p_tens = torch.FloatTensor(section_p).to(args.device)
    # print('Computing marginals...')
    # with torch.no_grad():
    #     token_embeddings, _ = model._compute_marginal(
    #         center_ids_tens, section_ids_tens.unsqueeze(1), section_p_tens.unsqueeze(1))
    #
    # token_embeddings = token_embeddings.squeeze(1)
    # center_words = [
    #     'cancer', 'patient', 'parking', 'advil', 'tylenol', 'pain', 'brain', 'daily', 'mri', 'x-ray', 'pleural',
    #     'systolic', 'nerve',
    # ]
    #
    # str = []
    # print('Computing similarity with respect to {}'.format(','.join(center_words)))
    # for center_word in center_words:
    #     center_id = token_vocab.get_id(center_word)
    #     mu = token_embeddings[center_id, :]
    #     sim = torch.nn.CosineSimilarity()(token_embeddings, mu.unsqueeze(0))
    #     top_word_ids = sim.topk(10).indices.cpu().numpy()
    #     top_tokens = [token_vocab.get_token(id) for id in top_word_ids]
    #     str.append('Closest words {} --> {}'.format(center_word, ', '.join(top_tokens)))
    #
    # results_dir = 'weights/{}/results'.format(args.experiment)
    # if not os.path.exists(results_dir):
    #     os.mkdir(results_dir)
    # # fp = os.path.join(results_dir, 'token_sim.txt')
    # # print('Saving token results to {}'.format(fp))
    # # with open(fp, 'w') as fd:
    # #     fd.write('\n'.join(str))
    #
    # metadata_embeddings = model.decoder.metadata_embeddings.weight
    # metadata = metadata_vocab.i2w[1:]
    # str = []
    # for section in metadata:
    #     sid = metadata_vocab.get_id(section)
    #     mu = metadata_embeddings[sid, :]
    #     sim = torch.nn.CosineSimilarity()(metadata_embeddings, mu.unsqueeze(0))
    #     top_section_ids = sim.topk(5).indices.cpu().numpy()
    #     top_sections = [metadata_vocab.get_token(id) for id in top_section_ids[1:]]
    #     str.append('Closest sections {} --> {}'.format(section, ', '.join(top_sections)))
    #
    # fp = os.path.join(results_dir, 'section_sim.txt')
    # print('Saving section results to {}'.format(fp))
    # with open(fp, 'w') as fd:
    #     fd.write('\n'.join(str))
