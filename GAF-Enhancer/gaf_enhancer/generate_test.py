import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader_gosai
import diffusion_gosai_update as diffusion_gosai
import utils
import random
import string
import datetime

import torch
import os
from pytorch_lightning import Callback
from torchvision.utils import save_image  # 可选，用于保存生成的图像
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from omegaconf import OmegaConf
# import ga
import diffusion_gosai_update
import numpy as np
import oracle
import torch
device = torch.device("cuda:0")

from torch.utils.data import DataLoader, TensorDataset
# DNA one-hot encoding class
class DNAoneHotEncoding:
    """
    DNA sequences one hot encoding
    """

    def __call__(self, sequence: str):
        assert (len(sequence) > 0)
        encoding = np.zeros((len(sequence), 4), dtype="float32")
        A = np.array([1, 0, 0, 0])
        C = np.array([0, 1, 0, 0])
        G = np.array([0, 0, 1, 0])
        T = np.array([0, 0, 0, 1])
        for index, nuc in enumerate(sequence):
            if nuc == "A":
                encoding[index, :] = A
            elif nuc == "C":
                encoding[index, :] = C
            elif nuc == "G":
                encoding[index, :] = G
            elif nuc == "T":
                encoding[index, :] = T
        return encoding

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import dataloader_gosai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import oracle
from scipy.stats import pearsonr
import torch
from tqdm import tqdm
import diffusion_gosai_cfg
from utils import set_seed
import json



set_seed(0, use_cuda=True)
base_path = ''
# our model
CKPT_PATH = ''

NUM_SAMPLE_BATCHES = 10
NUM_SAMPLES_PER_BATCH = 200
# reinitialize Hydra
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration|
initialize(config_path="configs_gosai", job_name="load_model")
cfg = compose(config_name="config_gosai.yaml")
cfg.eval.checkpoint_path = CKPT_PATH

# model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)

model = diffusion_gosai_update.Diffusion(cfg, eval=False).cuda()
model.load_state_dict(torch.load(cfg.eval.checkpoint_path))

model.eval()
all_detoeknized_samples = []
all_raw_samples = []
for _ in tqdm(range(NUM_SAMPLE_BATCHES)):
    samples = model._sample(eval_sp_size=NUM_SAMPLES_PER_BATCH)
    all_raw_samples.append(samples)
    detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
    all_detoeknized_samples.extend(detokenized_samples)
all_raw_samples = torch.concat(all_raw_samples)
# 存入 JSON 文件
with open("/drake_model_dna.json", "w") as f:
    json.dump(all_detoeknized_samples, f)
# 读取 JSON 文件
with open("/drake_model_dna.json", "r") as f:
    loaded_data = json.load(f)

n = len(loaded_data) // 4
all_detoeknized_samples1 = loaded_data[0:n]
all_detoeknized_samples2 = loaded_data[n:2*n]
all_detoeknized_samples3 = loaded_data[2*n:3*n]
all_detoeknized_samples4 = loaded_data[3*n:]
preds = []
for i in [all_detoeknized_samples1, all_detoeknized_samples2, all_detoeknized_samples3, all_detoeknized_samples4]:
    preds.append(oracle.cal_gosai_pred_new(i, mode='eval'))


generated_preds = np.concatenate(preds, axis=0)

sort = np.argsort(generated_preds)
fitness = generated_preds[sort][::-1]
np.savetxt("drake_model_dna_fitness.txt", generated_preds, fmt="%s", delimiter="\n")# epoch 10: 0.33 epoch 100+rl:
print()