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
import ga
import diffusion_gosai_update
import numpy as np



# import wandb
omegaconf.OmegaConf.register_new_resolver("uuid", lambda: ''.join(random.choice(string.ascii_letters) for _ in range(10))+'_'+str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
  if 'hf' in config.backbone:
    return diffusion_gosai.Diffusion(
      config, 
      ).to('cuda')
  
  return diffusion_gosai.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, test_ds):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds), ('test', test_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch seqs.shape', batch['seqs'].shape)
    print(f'tokens:', dataloader_gosai.dna_detokenize(batch['seqs'][0]))
    print('ids:', batch['seqs'][0])
    

def _train(config, logger):
  logger.info('Starting Training.')
  # wandb_logger = None
  # wandb_settings = wandb.Settings(
  #     base_url='https://api.wandb.ai'  # Specify your wandb host URL here
  # )
  # if config.get('wandb', None) is not None and not config.debug_mode:
  #   wandb_logger = L.pytorch.loggers.WandbLogger(
  #     config=omegaconf.OmegaConf.to_object(config),
  #     settings=wandb_settings,
  #     ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  # callbacks = []
  # if 'callbacks' in config:
  #   for _, callback in config.callbacks.items():
  #     callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds, test_ds = dataloader_gosai.get_dataloaders_gosai(config)

  model = diffusion_gosai.Diffusion(
    config, 
    )

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    strategy=hydra.utils.instantiate(config.strategy))
    # logger=wandb_logger)
  print('Start training...')
  # trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  trainer.fit(model, train_ds, ckpt_path=ckpt_path)

def _ga_train(config, logger, file):
  logger.info('Starting Training.')
  # wandb_logger = None
  # wandb_settings = wandb.Settings(
  #     base_url='https://api.wandb.ai'  # Specify your wandb host URL here
  # )
  # if config.get('wandb', None) is not None and not config.debug_mode:
  #   wandb_logger = L.pytorch.loggers.WandbLogger(
  #     config=omegaconf.OmegaConf.to_object(config),
  #     settings=wandb_settings,
  #     ** config.wandb)

  # if (config.checkpointing.resume_from_ckpt
  #     and config.checkpointing.resume_ckpt_path is not None
  #     and utils.fsspec_exists(
  #       config.checkpointing.resume_ckpt_path)):
  #   ckpt_path = config.checkpointing.resume_ckpt_path
  # else:
  #   ckpt_path = None

  # Lightning callbacks
  # callbacks = []
  # if 'callbacks' in config:
  #   for _, callback in config.callbacks.items():
  #     callbacks.append(hydra.utils.instantiate(callback))
  # CKPT_PATH = os.path.join(args.base_path, 'mdlm/outputs_gosai/pretrained.ckpt')

  if file != None:
    dna_sequences = []
    initialize(config_path="configs_gosai", job_name="load_model")
    cfg = compose(config_name="config_gosai.yaml")
    new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint('', config=cfg) # file path
    for _ in range(100):
      sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = new_model._sample_finetune_gradient(
        eval_sp_size=50, copy_flag_temp=None)  # {'A': 0, 'C': 1, 'G': 2, 'T': 3}
      index_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
      # base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
      dna_sequence = ["".join(index_to_base[i.item()] for i in seq) for seq in sample.argmax(dim=-1)]
      dna_sequences.extend(dna_sequence)
    # dna_sequences = [list(seq) for seq in dna_sequences]

    ga_seq, ga_fitness = ga.train(dna_sequences)
    sorted_indices = np.argsort(ga_fitness)[::-1]  # 排序并逆序，得到按适应度降序的索引
    top_indices = sorted_indices[:10000]
    ga_seq = [ga_seq[i] for i in top_indices]
    ga_fitness = [ga_fitness[i] for i in top_indices]

    train_ds, valid_ds, test_ds = dataloader_gosai.ga_dataloaders(config, ga_seq, ga_fitness)
  #
  else:
    # 无修改的训练
    train_ds, valid_ds, test_ds = dataloader_gosai.get_dataloaders_gosai(config)

  # 训练步骤
  model = diffusion_gosai.Diffusion(
    config,
  )
  # default_root_dir = os.getcwd()

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir='',# result path
    strategy=hydra.utils.instantiate(config.strategy)
  )
    # logger=wandb_logger)
  print('Start training...')


  # trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  trainer.fit(model, train_ds, ckpt_path=file)


@hydra.main(version_base=None, config_path='configs_gosai',
            config_name='config_gosai')
def main(config: DictConfig):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  logger = utils.get_logger(__name__)
  assert config.mode == 'train'
  file = config.checkpointing.resume_ckpt_path
  _ga_train(config, logger, file)

def modify_config_before_main(i):

  with initialize(config_path="configs_gosai"):

    config = compose(config_name="config_gosai")
    if i == 0:
      config.checkpointing.resume_ckpt_path = None
    else:
      config.checkpointing.resume_ckpt_path = f'//version_{i - 1}/checkpoints/epoch={(i - 1) * 10 + 9}-step={3930 + (i - 1) * 3538}.ckpt'
    return config


if __name__ == '__main__':
  import os

  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  import resource
  from hydra.core.global_hydra import GlobalHydra




  resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
  for i in range(10):
    if GlobalHydra.instance().is_initialized():
      GlobalHydra.instance().clear()
    print(f'{i+1}')
    config = modify_config_before_main(i)
    main(config)
    torch.cuda.empty_cache()

