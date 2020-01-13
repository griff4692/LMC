import argparse
import numpy as np
import pandas as pd
import torch

from lmc_utils import restore_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian LMC Skip-Gram Model')

    # Functional Arguments
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')
    args = parser.parse_args()

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    sfs = pd.read_csv('../eval/eval_data/minnesota/preprocessed_dataset_window_10.csv')['sf'].unique().tolist()
    sfs = list(map(lambda x: x.lower(), sfs))

    prev_args, model, token_vocab, section_vocab, optimizer_state, token_section_counts = restore_model(args.experiment)
    model.eval()
    model.to(args.device)

    top_n = 10000
    top_word_ids = list(range(top_n))
    section_ids = np.zeros([top_n,  section_vocab.size()])
    section_p = np.zeros([top_n, section_vocab.size()])
    for i in top_word_ids:
        if i > 0:
            sids, sp = token_section_counts[i]
            section_ids[i, :len(sids)] = sids
            section_p[i, :len(sp)] = sp

    center_ids_tens = torch.LongTensor(top_word_ids).unsqueeze(-1).to(args.device)
    section_ids_tens = torch.LongTensor(section_ids).to(args.device)
    section_p_tens = torch.FloatTensor(section_p).to(args.device)
    with torch.no_grad():
        token_embeddings, _ = model._compute_marginal(
            center_ids_tens, section_ids_tens.unsqueeze(1), section_p_tens.unsqueeze(1))

    token_embeddings = token_embeddings.squeeze(1)
    center_words = [
        'cancer', 'patient', 'parking', 'advil', 'tylenol', 'pain', 'brain', 'daily', 'mri', 'x-ray', 'pleural',
        'systolic', 'nerve',
    ]

    str = []
    for center_word in center_words:
        center_id = token_vocab.get_id(center_word)
        mu = token_embeddings[center_id, :]
        sim = torch.nn.CosineSimilarity()(token_embeddings, mu.unsqueeze(0))
        top_word_ids = sim.topk(10).indices.cpu().numpy()
        top_tokens = [token_vocab.get_token(id) for id in top_word_ids]
        str.append('Closest words {} --> {}'.format(center_word, ', '.join(top_tokens)))
    fp = 'weights/{}/results/token_sim.txt'.format(args.experiment)
    print('Saving token results to {}'.format(fp))
    with open(fp, 'w') as fd:
        fd.write('\n'.join(str))

    section_embeddings = model.encoder.section_embeddings.weight
    sections = section_vocab.i2w[1:]
    str = []
    for section in sections:
        sid = section_vocab.get_id(section)
        mu = section_embeddings[sid, :]
        sim = torch.nn.CosineSimilarity()(section_embeddings, mu.unsqueeze(0))
        top_section_ids = sim.topk(5).indices.cpu().numpy()
        top_sections = [section_vocab.get_token(id) for id in top_section_ids[1:]]
        str.append('Closest sections {} --> {}'.format(section, ', '.join(top_sections)))

    fp = 'weights/{}/results/section_sim.txt'.format(args.experiment)
    print('Saving section results to {}'.format(fp))
    with open(fp, 'w') as fd:
        fd.write('\n'.join(str))
