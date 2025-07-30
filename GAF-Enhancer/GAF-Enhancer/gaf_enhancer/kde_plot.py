
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')


with open("/wgan_dna.json", "r") as f:
    wgan_dna = json.load(f)
wgan_fitness = np.loadtxt("/wgan_fitness.txt", dtype=float)

with open("/drake_model_dna_first.json", "r") as f:
    drake_model_dna_first = json.load(f)
drake_model_dna_first_fitness = np.loadtxt("/drake_model_dna_first_fitness.txt", dtype=float)

with open("/drake_model_dna.json", "r") as f:
    drake_model_dna = json.load(f)
drake_model_dna_fitness = np.loadtxt("/drake_model_dna_fitness.txt", dtype=float)

with open("/best_model_dna_first.json", "r") as f:
    best_model_dna_first = json.load(f)
best_model_dna_first_fitness = np.loadtxt("/best_model_dna_first_fitness.txt", dtype=float)

with open("/best_model_dna.json", "r") as f:
    best_model_dna = json.load(f)
best_model_dna_fitness = np.loadtxt("/best_model_dna_fitness.txt", dtype=float)


plt.rcParams.update({
    'font.sans-serif': 'Times New Roman',
})
plt.rcParams['font.size'] = 20


fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 改为1×4布局

sns.kdeplot(drake_model_dna_first_fitness, label='Group 1', shade=True, alpha=0.5, ax=axes[0])
axes[0].set_title('GAF-Enhancer(no-feedback-reinforce)')
axes[0].set_xlim(-3, 9)
axes[0].set_ylim(0, 0.9)
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')


zoom_in_1 = axes[0].inset_axes([0.6, 0.6, 0.3, 0.3])
sns.kdeplot(drake_model_dna_first_fitness, shade=True, alpha=0.5, ax=zoom_in_1)
zoom_in_1.set_xlim(7, 9)
zoom_in_1.set_ylim(0, 0.05)
zoom_in_1.set_xticks([])
zoom_in_1.set_yticks([])


sns.kdeplot(drake_model_dna_fitness, label='Group 2', shade=True, alpha=0.5, ax=axes[1])
axes[1].set_title('GAF-Enhancer(no-feedback)')
axes[1].set_xlim(-3, 9)
axes[1].set_ylim(0, 0.9)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')



sns.kdeplot(best_model_dna_first_fitness, label='Group 3', shade=True, alpha=0.5, ax=axes[2])
axes[2].set_title('GAF-Enhancer(no-reinforce)')
axes[2].set_xlim(-3, 9)
axes[2].set_ylim(0, 0.9)
axes[2].set_xlabel('Value')
axes[2].set_ylabel('')


zoom_in_2 = axes[2].inset_axes([0.6, 0.6, 0.3, 0.3])
sns.kdeplot(best_model_dna_first_fitness, shade=True, alpha=0.5, ax=zoom_in_2)
zoom_in_2.set_xlim(7, 9)
zoom_in_2.set_ylim(0, 0.05)
zoom_in_2.set_xticks([])
zoom_in_2.set_yticks([])


sns.kdeplot(best_model_dna_fitness, label='Group 4', shade=True, alpha=0.5, ax=axes[3])
axes[3].set_title('GAF-Enhancer')
axes[3].set_xlim(-3, 9)
axes[3].set_ylim(0, 0.9)
axes[3].set_xlabel('Value')
axes[3].set_ylabel('')


plt.tight_layout()


plt.savefig('kde_plot_1x4_600dpi.png', dpi=600)


plt.show()

