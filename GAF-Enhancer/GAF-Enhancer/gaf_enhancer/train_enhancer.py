# import wandb
import os
os.environ["WANDB_MODE"] = "disabled"
import grelu
import pandas as pd
from grelu.lightning import LightningModel
import grelu.data.dataset
import os

base_path = ''
df_train = pd.read_csv(f'/real_Sequence_activity_train.txt', sep='\t')
df_val = pd.read_csv(f'/real_Sequence_activity_val.txt', sep='\t')

model_params = {
    'model_type':'EnformerPretrainedModel',
    'n_tasks': 1,
}

train_params = {
    'task':'regression',
    'loss': 'MSE',
    'lr':1e-4,
    # 'logger': 'wandb',
    'batch_size': 512,
    'num_workers': 4,
    'devices': [1],
    'save_dir': os.path.join(base_path, 'mdlm/outputs_gosai'),
    'optimizer': 'adam',
    'max_epochs': 10,
    'checkpoint': True,
}

def train_model():
    train_data = df_train[['sequence', 'Dev_activity_log2']]
    val_data = df_val[['sequence', 'Dev_activity_log2']]

    # Make labeled datasets
    train_dataset = grelu.data.dataset.DFSeqDataset(train_data)
    val_dataset   = grelu.data.dataset.DFSeqDataset(val_data)

    model = LightningModel(model_params=model_params, train_params=train_params)
    trainer = model.train_on_dataset(train_dataset, val_dataset)
    # wandb.finish()
    return model

model = train_model()
