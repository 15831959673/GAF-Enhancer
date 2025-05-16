import numpy as np
import matplotlib.pyplot as plt
import ptitprince as pt  # 需要安装 ptitprince: pip install ptitprince
import  random
import json
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
import matplotlib.pyplot as plt

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
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first.json", "r") as f:
    top_dna = json.load(f)
top_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/top_dna_fitness.txt", dtype=float)

# real_top
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna.json", "r") as f:
    best_model_dna = json.load(f)
best_model_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_fitness.txt", dtype=float)

print(f'avg: {np.mean(wgan_fitness):.3g}, '
      f'{np.mean(drake_model_dna_first_fitness):.3g}, '
      f'{np.mean(drake_model_dna_fitness):.3g}, '
      f'{np.mean(best_model_dna_first_fitness):.3g}, '
      f'{np.mean(best_model_dna_fitness):.3g}')

# 设置字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# 获取 Top-50 和 Top-100

drake_model_dna_fitness = np.sort(drake_model_dna_fitness)[::-1]
best_model_dna_fitness = np.sort(best_model_dna_fitness)[::-1]

top_50_1 = wgan_fitness[:50]
top_100_1 = wgan_fitness[:100]
top_50_2 = drake_model_dna_fitness[:50]
top_100_2 = drake_model_dna_fitness[:100]
top_50_3 = best_model_dna_fitness[:50]
top_100_3 = best_model_dna_fitness[:100]

# 组合数据
data_for_plot = [
    (top_50_1, 'Top-50 W-GAN'),
    (top_100_1, 'Top-100 W-GAN'),
    (wgan_fitness, '2000 W-GAN'),
    (top_50_2, 'Top-50 DRAKES'),
    (top_100_2, 'Top-100 DRAKES'),
    (drake_model_dna_fitness, '2000 DRAKES'),
    (top_50_3, 'Top-50 GAF-Enhancer'),
    (top_100_3, 'Top-100 GAF-Enhancer'),
    (best_model_dna_fitness, '2000 GAF-Enhancer')
]

# 转换为 DataFrame
import pandas as pd
df = pd.concat([
    pd.DataFrame({'Values': data, 'Category': label}) for data, label in data_for_plot
])
# 绘制云雨图
plt.figure(figsize=(10, 4))
ax = pt.RainCloud(x='Category', y='Values', data=df, palette='Set2', width_viol=0.7, width_box=0.2, point_size=2, alpha=0.6)

ax.set_xlabel('')  # 去除 x 轴的 "Category" 标签

plt.xticks(rotation=30)
# plt.title('Raincloud Plot of Top-50 and Top-100 Values from Three Datasets')
# plt.xlabel('Datasets and Top-N')
plt.ylabel('Values', fontsize=12)
plt.ylim(-2, 10)
plt.tight_layout()

# 保存为图片
plt.savefig('raincloud_plot600.png', dpi=600)
plt.show()
