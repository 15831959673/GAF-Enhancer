# import seaborn as sns
# import matplotlib.pyplot as plt
# import json
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
#
# # wgan
# with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/wgan_dna.json", "r") as f:
#     wgan_dna = json.load(f)
# wgan_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/wgan_fitness.txt", dtype=float)
#
# # drake_first
# with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_first.json", "r") as f:
#     drake_model_dna_first = json.load(f)
# drake_model_dna_first_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_first_fitness.txt", dtype=float)
#
# # drake
# with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna.json", "r") as f:
#     drake_model_dna = json.load(f)
# drake_model_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_fitness.txt", dtype=float)
#
# # best_first
# with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first.json", "r") as f:
#     best_model_dna_first = json.load(f)
# best_model_dna_first_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first_fitness.txt", dtype=float)
#
# # best
# with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first.json", "r") as f:
#     top_dna = json.load(f)
# top_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/top_dna_fitness.txt", dtype=float)
#
# # real_top
# with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna.json", "r") as f:
#     best_model_dna = json.load(f)
# best_model_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_fitness.txt", dtype=float)
#
# # 创建 2x2 的子图布局
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# # 设置字体
# # 设置全局字体
# plt.rcParams.update({
#     'font.sans-serif': 'Times New Roman',  # 设置字体类型
#     # 'font.size': 16,  # 全局字体大小
#     # 'axes.titlesize': 16,  # 子图标题字体大小
#     # 'axes.labelsize': 14,  # 坐标轴标签字体大小
#     # 'xtick.labelsize': 14,  # x轴刻度字体大小
#     # 'ytick.labelsize': 14,  # y轴刻度字体大小
#     # 'legend.fontsize': 14  # 图例字体大小
# })
#
#
# # 绘制核密度估计图
# sns.kdeplot(drake_model_dna_first_fitness, label='Group 1', shade=True, alpha=0.5, ax=axes[0, 0])
# axes[0, 0].set_title('GAF-Enhancer(no-feedback-reinforce)', fontsize=16)
# axes[0, 0].set_xlim(-3, 9)
# axes[0, 0].set_ylim(0, 0.9)
# axes[0, 0].set_xlabel('Value', fontsize=16)
# axes[0, 0].set_ylabel('Density', fontsize=16)
#
# sns.kdeplot(drake_model_dna_fitness, label='Group 2', shade=True, alpha=0.5, ax=axes[0, 1])
# axes[0, 1].set_title('GAF-Enhancer(no-feedback)', fontsize=16)
# axes[0, 1].set_xlim(-3, 9)
# axes[0, 1].set_ylim(0, 0.9)
# axes[0, 1].set_xlabel('Value', fontsize=16)
# axes[0, 1].set_ylabel('Density', fontsize=16)
#
# sns.kdeplot(best_model_dna_first_fitness, label='Group 3', shade=True, alpha=0.5, ax=axes[1, 0])
# axes[1, 0].set_title('GAF-Enhancer(no-reinforce)', fontsize=16)
# axes[1, 0].set_xlim(-3, 9)
# axes[1, 0].set_ylim(0, 0.9)
# axes[1, 0].set_xlabel('Value', fontsize=16)
# axes[1, 0].set_ylabel('Density', fontsize=16)
#
# sns.kdeplot(best_model_dna_fitness, label='Group 4', shade=True, alpha=0.5, ax=axes[1, 1])
# axes[1, 1].set_title('GAF-Enhancer', fontsize=16)
# axes[1, 1].set_xlim(-3, 9)
# axes[1, 1].set_ylim(0, 0.9)
# axes[1, 1].set_xlabel('Value', fontsize=16)
# axes[1, 1].set_ylabel('Density', fontsize=16)
#
# zoom_in_1 = axes[0, 0].inset_axes([0.6, 0.6, 0.3, 0.3])  # [x位置, y位置, 宽度, 高度]
# sns.kdeplot(drake_model_dna_first_fitness, shade=True, alpha=0.5, ax=zoom_in_1)
# zoom_in_1.set_xlim(7, 9)  # 设置放大区域x轴范围
# zoom_in_1.set_ylim(0, 0.05)  # 设置放大区域y轴范围
# zoom_in_1.set_xticks([])  # 去掉x轴刻度
# zoom_in_1.set_yticks([])  # 去掉y轴刻度
#
# # 对于第三个子图
# zoom_in_2 = axes[1, 0].inset_axes([0.6, 0.6, 0.3, 0.3])  # [x位置, y位置, 宽度, 高度]
# sns.kdeplot(best_model_dna_first_fitness, shade=True, alpha=0.5, ax=zoom_in_2)
# zoom_in_2.set_xlim(7, 9)  # 设置放大区域x轴范围
# zoom_in_2.set_ylim(0, 0.05)  # 设置放大区域y轴范围
# zoom_in_2.set_xticks([])  # 去掉x轴刻度
# zoom_in_2.set_yticks([])  # 去掉y轴刻度
# # 调整布局
# plt.tight_layout()
#
# # 保存为图片
# plt.savefig('kde_plot_subplots600.png', dpi=600)
#
# # 显示图表
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

# 数据加载
with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/wgan_dna.json", "r") as f:
    wgan_dna = json.load(f)
wgan_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/wgan_fitness.txt", dtype=float)

with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_first.json", "r") as f:
    drake_model_dna_first = json.load(f)
drake_model_dna_first_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_first_fitness.txt", dtype=float)

with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna.json", "r") as f:
    drake_model_dna = json.load(f)
drake_model_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/drake_model_dna_fitness.txt", dtype=float)

with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first.json", "r") as f:
    best_model_dna_first = json.load(f)
best_model_dna_first_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_first_fitness.txt", dtype=float)

with open("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna.json", "r") as f:
    best_model_dna = json.load(f)
best_model_dna_fitness = np.loadtxt("/home/liangwei/zhl/DRAKES_lw/drakes_dna/best_model_dna_fitness.txt", dtype=float)

# 设置字体
plt.rcParams.update({
    'font.sans-serif': 'Times New Roman',
})
plt.rcParams['font.size'] = 20

# 创建 1x4 子图布局
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 改为1×4布局

# 子图1
sns.kdeplot(drake_model_dna_first_fitness, label='Group 1', shade=True, alpha=0.5, ax=axes[0])
axes[0].set_title('GAF-Enhancer(no-feedback-reinforce)')
axes[0].set_xlim(-3, 9)
axes[0].set_ylim(0, 0.9)
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')

# 放大图1
zoom_in_1 = axes[0].inset_axes([0.6, 0.6, 0.3, 0.3])
sns.kdeplot(drake_model_dna_first_fitness, shade=True, alpha=0.5, ax=zoom_in_1)
zoom_in_1.set_xlim(7, 9)
zoom_in_1.set_ylim(0, 0.05)
zoom_in_1.set_xticks([])
zoom_in_1.set_yticks([])

# 子图2
sns.kdeplot(drake_model_dna_fitness, label='Group 2', shade=True, alpha=0.5, ax=axes[1])
axes[1].set_title('GAF-Enhancer(no-feedback)')
axes[1].set_xlim(-3, 9)
axes[1].set_ylim(0, 0.9)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')


# 子图3
sns.kdeplot(best_model_dna_first_fitness, label='Group 3', shade=True, alpha=0.5, ax=axes[2])
axes[2].set_title('GAF-Enhancer(no-reinforce)')
axes[2].set_xlim(-3, 9)
axes[2].set_ylim(0, 0.9)
axes[2].set_xlabel('Value')
axes[2].set_ylabel('')

# 放大图3
zoom_in_2 = axes[2].inset_axes([0.6, 0.6, 0.3, 0.3])
sns.kdeplot(best_model_dna_first_fitness, shade=True, alpha=0.5, ax=zoom_in_2)
zoom_in_2.set_xlim(7, 9)
zoom_in_2.set_ylim(0, 0.05)
zoom_in_2.set_xticks([])
zoom_in_2.set_yticks([])

# 子图4
sns.kdeplot(best_model_dna_fitness, label='Group 4', shade=True, alpha=0.5, ax=axes[3])
axes[3].set_title('GAF-Enhancer')
axes[3].set_xlim(-3, 9)
axes[3].set_ylim(0, 0.9)
axes[3].set_xlabel('Value')
axes[3].set_ylabel('')

# 自动调整布局
plt.tight_layout()

# 保存图像
plt.savefig('kde_plot_1x4_600dpi.png', dpi=600)

# 显示图表
plt.show()

