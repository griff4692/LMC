import pickle

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import tensor_to_np

import torch


if __name__ == '__main__':
    checkpoint_fp = '../model/weights/lga/checkpoint_4.pth'

    if not torch.cuda.is_available():
        checkpoint_state = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(checkpoint_fp)
    section_vocab = checkpoint_state['section_vocab']
    embeddings = tensor_to_np(checkpoint_state['model_state_dict']['encoder.section_embeddings.weight'])

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    sections = section_vocab.i2w
    sections = [s.split('=')[-1][:-1] for s in sections]

    sections_df = pd.read_csv('../preprocess/data/mimic/sections.csv')
    common_sections = set(sections_df[sections_df['count'] >= 25000]['section'].tolist())
    common_sections = common_sections.intersection(set(sections))

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
        ax.scatter(row['x'], row['y'], label=row['section'], alpha=0.3, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.show()

