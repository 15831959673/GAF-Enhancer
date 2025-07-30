# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
import matplotlib
matplotlib.use('Agg')



def one_hot_encode(dna_sequence):

    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    encoded = np.zeros((len(dna_sequence), 4))
    for i, char in enumerate(dna_sequence):
        encoded[i, char_to_int[char]] = 1
    return encoded.flatten()

# wgan
with open("/wgan_dna.json", "r") as f:
    wgan_dna = json.load(f)
wgan_fitness = np.loadtxt("/wgan_fitness.txt", dtype=float)

# drake_first
with open("/drake_model_dna_first.json", "r") as f:
    drake_model_dna_first = json.load(f)
drake_model_dna_first_fitness = np.loadtxt("/drake_model_dna_first_fitness.txt", dtype=float)

# drake
with open("/drake_model_dna.json", "r") as f:
    drake_model_dna = json.load(f)
drake_model_dna_fitness = np.loadtxt("/drake_model_dna_fitness.txt", dtype=float)

# best_first
with open("/best_model_dna_first.json", "r") as f:
    best_model_dna_first = json.load(f)
best_model_dna_first_fitness = np.loadtxt("/best_model_dna_first_fitness.txt", dtype=float)

# best
with open("/top_dna.json", "r") as f:
    top_dna = json.load(f)
top_dna_fitness = np.loadtxt("/top_dna_fitness.txt", dtype=float)

# real_top
with open("/best_model_dna.json", "r") as f:
    best_model_dna = json.load(f)
best_model_dna_fitness = np.loadtxt("/best_model_dna_fitness.txt", dtype=float)

import numpy as np
import pandas as pd
import random


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] =22


df = pd.read_csv('/real_Sequence_activity_train.txt', sep='\t')


dna_sequences = df['sequence'].tolist()  #
activity_values = df[['Dev_activity_log2']].values


random_indices = random.sample(range(len(dna_sequences)), 10000)
random_dna_sequences = [dna_sequences[i] for i in random_indices]
random_activity_values = activity_values[random_indices].squeeze()

dna = best_model_dna + random_dna_sequences
dna_fitness = np.concatenate((best_model_dna_fitness, random_activity_values))

dna1 = best_model_dna_first +random_dna_sequences
dna_fitness1 = np.concatenate((best_model_dna_first_fitness, random_activity_values))

one_hot_sequences1 = np.array([one_hot_encode(seq) for seq in random_dna_sequences])
one_hot_sequences2 = np.array([one_hot_encode(seq) for seq in best_model_dna])
one_hot_sequences3 = np.array([one_hot_encode(seq) for seq in dna])
one_hot_sequences4 = np.array([one_hot_encode(seq) for seq in best_model_dna_first])
one_hot_sequences5 = np.array([one_hot_encode(seq) for seq in dna1])
one_hot_sequences6 = np.array([one_hot_encode(seq) for seq in drake_model_dna_first])


tsne = TSNE(n_components=2, random_state=42)
tsne_result1 = tsne.fit_transform(one_hot_sequences1)
tsne_result2 = tsne.fit_transform(one_hot_sequences2)
tsne_result3 = tsne.fit_transform(one_hot_sequences3)
tsne_result4 = tsne.fit_transform(one_hot_sequences4)
tsne_result5 = tsne.fit_transform(one_hot_sequences5)
tsne_result6 = tsne.fit_transform(one_hot_sequences6)

# norm = plt.Normalize(vmin=np.min(best_model_dna_fitness), vmax=np.max(best_model_dna_fitness))


fig, axes = plt.subplots(2, 3, figsize=(24, 12), gridspec_kw={'width_ratios': [1.2, 1.2, 1.2]})  # 1行3列


scatter1 = axes[0, 0].scatter(tsne_result6[:, 0], tsne_result6[:, 1], c=drake_model_dna_first_fitness, s=50, cmap='Blues', alpha=0.2)
axes[0, 0].set_title('2000 first phase samples no feedback')
axes[0, 0].text(0.05, 0.95, '(a)', transform=axes[0, 0].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter1, ax=axes[0, 0], label='Activity Value')

scatter2 = axes[0, 1].scatter(tsne_result4[:, 0], tsne_result4[:, 1], c=best_model_dna_first_fitness, s=50, cmap='Blues', alpha=0.2)
axes[0, 1].set_title('2000 first phase samples with feedback')
axes[0, 1].text(0.05, 0.95, '(b)', transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter2, ax=axes[0, 1], label='Activity Value')

scatter3 = axes[0, 2].scatter(tsne_result2[:, 0], tsne_result2[:, 1], c=best_model_dna_fitness, s=50, cmap='Blues', alpha=0.2)
axes[0, 2].set_title('2000 second phase samples')
axes[0, 2].text(0.05, 0.95, '(c)', transform=axes[0, 2].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[0, 2], label='Activity Value')

scatter3 = axes[1, 0].scatter(tsne_result1[:, 0], tsne_result1[:, 1], c=random_activity_values, s=50, cmap='Blues', alpha=0.2)
axes[1, 0].set_title('10000 real samples')
axes[1, 0].text(0.05, 0.95, '(d)', transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[1, 0], label='Activity Value')

scatter3 = axes[1, 1].scatter(tsne_result5[:, 0], tsne_result5[:, 1], c=dna_fitness1, s=50, cmap='Blues', alpha=0.2)
axes[1, 1].set_title('10000 real samples + 2000 first phase samples')
axes[1, 1].text(0.05, 0.95, '(e)', transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[1, 1], label='Activity Value')

scatter3 = axes[1, 2].scatter(tsne_result3[:, 0], tsne_result3[:, 1], c=dna_fitness, s=50, cmap='Blues', alpha=0.2)
axes[1, 2].set_title('10000 real samples + 2000 second phase samples')
axes[1, 2].text(0.05, 0.95, '(f)', transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[1, 2], label='Activity Value')


plt.tight_layout()


plt.savefig('tsne_subplots_with_labels600.pdf', dpi=600)
plt.show()

