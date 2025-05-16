# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
import matplotlib
matplotlib.use('Agg')

# 假设 dna_sequences 是一个包含2000个DNA序列的list
# 假设 activity_values 是一个包含2000个活性值的array

# 1. DNA序列的One-hot编码
def one_hot_encode(dna_sequence):
    # 定义字母到索引的映射
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # 初始化一个全0的矩阵
    encoded = np.zeros((len(dna_sequence), 4))
    for i, char in enumerate(dna_sequence):
        encoded[i, char_to_int[char]] = 1
    return encoded.flatten()

# wgan
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/wgan_dna.json", "r") as f:
    wgan_dna = json.load(f)
wgan_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/wgan_fitness.txt", dtype=float)

# drake_first
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_first.json", "r") as f:
    drake_model_dna_first = json.load(f)
drake_model_dna_first_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_first_fitness.txt", dtype=float)

# drake
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna.json", "r") as f:
    drake_model_dna = json.load(f)
drake_model_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_fitness.txt", dtype=float)

# best_first
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first.json", "r") as f:
    best_model_dna_first = json.load(f)
best_model_dna_first_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first_fitness.txt", dtype=float)

# best
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/top_dna.json", "r") as f:
    top_dna = json.load(f)
top_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/top_dna_fitness.txt", dtype=float)

# real_top
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna.json", "r") as f:
    best_model_dna = json.load(f)
best_model_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_fitness.txt", dtype=float)

import numpy as np
import pandas as pd
import random

# 设置字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] =22

# 读取txt文件（假设文件是以制表符分隔的）
# 可以根据实际情况调整分隔符（例如 ',' 或 '\t'）
df = pd.read_csv('/home/liangwei/zhl/DRAKES_lw/enhancer_250_EnhanerGAN/real_Sequence_activity_train.txt', sep='\t')

# 提取DNA序列和活性值
dna_sequences = df['sequence'].tolist()  # 将DNA序列提取为list
activity_values = df[['Dev_activity_log2']].values  # 提取活性值为array

# 随机选择10000条数据
random_indices = random.sample(range(len(dna_sequences)), 10000)
random_dna_sequences = [dna_sequences[i] for i in random_indices]
random_activity_values = activity_values[random_indices].squeeze()

dna = best_model_dna + random_dna_sequences
dna_fitness = np.concatenate((best_model_dna_fitness, random_activity_values))

dna1 = best_model_dna_first +random_dna_sequences
dna_fitness1 = np.concatenate((best_model_dna_first_fitness, random_activity_values))

# 将所有序列转化为One-hot编码
one_hot_sequences1 = np.array([one_hot_encode(seq) for seq in random_dna_sequences])
one_hot_sequences2 = np.array([one_hot_encode(seq) for seq in best_model_dna])
one_hot_sequences3 = np.array([one_hot_encode(seq) for seq in dna])
one_hot_sequences4 = np.array([one_hot_encode(seq) for seq in best_model_dna_first])
one_hot_sequences5 = np.array([one_hot_encode(seq) for seq in dna1])
one_hot_sequences6 = np.array([one_hot_encode(seq) for seq in drake_model_dna_first])


# 2. 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
tsne_result1 = tsne.fit_transform(one_hot_sequences1)
tsne_result2 = tsne.fit_transform(one_hot_sequences2)
tsne_result3 = tsne.fit_transform(one_hot_sequences3)
tsne_result4 = tsne.fit_transform(one_hot_sequences4)
tsne_result5 = tsne.fit_transform(one_hot_sequences5)
tsne_result6 = tsne.fit_transform(one_hot_sequences6)

# 3. 根据活性值的大小进行颜色映射
# 使用活性值进行颜色映射
# norm = plt.Normalize(vmin=np.min(best_model_dna_fitness), vmax=np.max(best_model_dna_fitness))


# 创建3个子图
fig, axes = plt.subplots(2, 3, figsize=(24, 12), gridspec_kw={'width_ratios': [1.2, 1.2, 1.2]})  # 1行3列

# 子图1
scatter1 = axes[0, 0].scatter(tsne_result6[:, 0], tsne_result6[:, 1], c=drake_model_dna_first_fitness, s=50, cmap='Blues', alpha=0.2)
axes[0, 0].set_title('2000 first phase samples no feedback')
axes[0, 0].text(0.05, 0.95, '(a)', transform=axes[0, 0].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter1, ax=axes[0, 0], label='Activity Value')

# 子图2
scatter2 = axes[0, 1].scatter(tsne_result4[:, 0], tsne_result4[:, 1], c=best_model_dna_first_fitness, s=50, cmap='Blues', alpha=0.2)
axes[0, 1].set_title('2000 first phase samples with feedback')
axes[0, 1].text(0.05, 0.95, '(b)', transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter2, ax=axes[0, 1], label='Activity Value')

# 子图3
scatter3 = axes[0, 2].scatter(tsne_result2[:, 0], tsne_result2[:, 1], c=best_model_dna_fitness, s=50, cmap='Blues', alpha=0.2)
axes[0, 2].set_title('2000 second phase samples')
axes[0, 2].text(0.05, 0.95, '(c)', transform=axes[0, 2].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[0, 2], label='Activity Value')

# 子图4
scatter3 = axes[1, 0].scatter(tsne_result1[:, 0], tsne_result1[:, 1], c=random_activity_values, s=50, cmap='Blues', alpha=0.2)
axes[1, 0].set_title('10000 real samples')
axes[1, 0].text(0.05, 0.95, '(d)', transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[1, 0], label='Activity Value')

# 子图5
scatter3 = axes[1, 1].scatter(tsne_result5[:, 0], tsne_result5[:, 1], c=dna_fitness1, s=50, cmap='Blues', alpha=0.2)
axes[1, 1].set_title('10000 real samples + 2000 first phase samples')
axes[1, 1].text(0.05, 0.95, '(e)', transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[1, 1], label='Activity Value')

# 子图6
scatter3 = axes[1, 2].scatter(tsne_result3[:, 0], tsne_result3[:, 1], c=dna_fitness, s=50, cmap='Blues', alpha=0.2)
axes[1, 2].set_title('10000 real samples + 2000 second phase samples')
axes[1, 2].text(0.05, 0.95, '(f)', transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=22, weight='bold', color='black')
plt.colorbar(scatter3, ax=axes[1, 2], label='Activity Value')

# 调整布局
plt.tight_layout()

# 保存和展示图像
plt.savefig('tsne_subplots_with_labels600.pdf', dpi=600)
plt.show()

