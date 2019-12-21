import re

import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model_utils import tensor_to_np

import torch


if __name__ == '__main__':
    is_lga = False

    fp_str = 'lga.png' if is_lga else 'sec2vec.png'
    title = 'Latent Meaning Cells' if is_lga else 'BSG With Headers As Pseudo-Contexts'

    if is_lga:
        checkpoint_fp = '../model/weights/lga/checkpoint_1.pth'
        if not torch.cuda.is_available():
            checkpoint_state = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)
        else:
            checkpoint_state = torch.load(checkpoint_fp)
        section_vocab = checkpoint_state['section_vocab']
        embeddings = tensor_to_np(checkpoint_state['model_state_dict']['encoder.section_embeddings.weight'])
        sections = section_vocab.i2w[1:]
    else:
        checkpoint_fp = '../model/weights/12-20-sec2vec/checkpoint_1.pth'
        if not torch.cuda.is_available():
            checkpoint_state = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)
        else:
            checkpoint_state = torch.load(checkpoint_fp)
        vocab = checkpoint_state['vocab']
        offset = vocab.separator_start_vocab_id
        embeddings = tensor_to_np(checkpoint_state['model_state_dict']['encoder.embeddings.weight'])[offset:, :]
        sections = vocab.i2w[offset:]
    tsne = TSNE(n_components=2, verbose=1, perplexity=50)
    tsne_results = tsne.fit_transform(embeddings)

    sections = [s.split('=')[1] for s in sections]
    sections_df = pd.read_csv('../preprocess/data/mimic/sections.csv')
    common_sections = sections_df[sections_df['count'] >= 45000]['section'].tolist()
    common_sections = set([re.sub('\s+', '', s).upper() for s in common_sections])
    x = []
    y = []
    for section in common_sections:
        sidx = sections.index(section)
        x.append(tsne_results[sidx, 0])
        y.append(tsne_results[sidx, 1])

    df = pd.DataFrame({
        'section': list(common_sections),
        'x': x,
        'y': y,
    })

    fig, ax = plt.subplots()
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        ax.scatter(row['x'], row['y'], s=150, label=row['section'], edgecolors='black')

    # ax.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # fig.subplots_adjust(bottom=0.2)
    ax.grid(True)
    plt.title('Section Header T-SNE Plots for {}'.format(title), fontdict={'fontsize': 9})
    plt.savefig('eval_data/{}'.format(fp_str))
    plt.show()
