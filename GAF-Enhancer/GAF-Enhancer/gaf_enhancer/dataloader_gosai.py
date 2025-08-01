import torch
import pandas as pd
import typing
import math
import utils
import numpy as np
import os

base_path = '' # base_path
LOGGER = utils.get_logger(__name__)
DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3} #, 'M': 4}
INDEX_TO_DNA = {v: k for k, v in DNA_ALPHABET.items()}
lookup_array = np.array([INDEX_TO_DNA[i] for i in range(len(INDEX_TO_DNA))])


def dna_to_tensor(dna_list):
  # 初始化一个列表来存储转换后的数据
  indexed_data = []

  for seq in dna_list:
    # 将序列中的每个字符映射到对应的索引
    indexed_seq = [DNA_ALPHABET[base] for base in seq]
    indexed_data.append(indexed_seq)

  # 转换为PyTorch tensor
  tensor_data = torch.tensor(indexed_data, dtype=torch.long)
  return tensor_data


def dna_detokenize(seq):
  return ''.join([list(DNA_ALPHABET.keys())[int(i)] for i in seq])

def batch_dna_detokenize(batch_seq):
    """
    batch_seq: numpy array of shape [batch_size, seq_len]
    return: list of strings
    """
    detokenized_batch = lookup_array[batch_seq]
    detokenized_batch = [''.join(seq) for seq in detokenized_batch]
    return detokenized_batch

def dna_tokenize(seq):
  return [DNA_ALPHABET[c] for c in seq]

def batch_dna_tokenize(batch_seq):
    """
    batch_seq: list of strings
    return: numpy array of shape [batch_size, seq_len]
    """
    tokenized_batch = np.array([[DNA_ALPHABET[c] for c in seq] for seq in batch_seq])
    return tokenized_batch

class GosaiDataset(torch.utils.data.Dataset):
    def __init__(self):
        data_df = pd.read_csv(os.path.join(base_path, f'mdlm/gosai_data/processed_data/gosai_all.csv'))
        self.seqs = torch.tensor(data_df['seq'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
        self.clss = torch.tensor(data_df[['hepg2', 'k562', 'sknsh']].to_numpy())
        LOGGER.info(f'Loaded data: seqs shape: {self.seqs.shape}, clss shape: {self.clss.shape}')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}

class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, name):
      df = pd.read_csv(f'/home/zhl/workspace/DRAKES/enhancer_250_EnhanerGAN/real_Sequence_activity_{name}.txt', sep='\t')
      # 转换DNA序列为One-Hot编码
      self.seqs = torch.tensor(df['sequence'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
      # 提取数值列
      self.clss = torch.tensor(df[['Dev_activity_log2']].values)
      LOGGER.info(f'Loaded data: seqs shape: {self.seqs.shape}, clss shape: {self.clss.shape}')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}


class GaDataset(torch.utils.data.Dataset):
  def __init__(self, name, ga_seq, ga_fitness):
    self.ga_seq =  dna_to_tensor(ga_seq)
    self.ga_fitness = torch.tensor(ga_fitness).unsqueeze(1).double()

    df = pd.read_csv(f'/enhancer_250_EnhanerGAN/real_Sequence_activity_{name}.txt', sep='\t') # dataset_path

    self.seqs = torch.tensor(df['sequence'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
    self.clss = torch.tensor(df[['Dev_activity_log2']].values)

    _, sorted_indices = torch.sort(self.clss, dim=0)  # 获取按适应度升序排序的索引
    sorted_indices = sorted_indices.squeeze(1)  # 将维度压缩为1D

    # 2. 获取适应度最小的 10000 个索引
    min_10000_indices = sorted_indices[:10000]  # 取最小的10000个索引
    # temp = self.clss[min_10000_indices]
    # 3. 用第一组数据替换第二组中的适应度最小的部分
    # 替换 tensor3（DNA序列的向量表示）和 tensor4（适应度值）
    self.seqs[min_10000_indices] = self.ga_seq  # 替换对应的 DNA 向量表示
    self.clss[min_10000_indices] = self.ga_fitness  # 替换对应的适应度值

    LOGGER.info(f'Loaded data: seqs shape: {self.seqs.shape}, clss shape: {self.clss.shape}')

  def __len__(self):
    return len(self.seqs)

  def __getitem__(self, idx):
    return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}



def get_datasets_gosai():
  return GosaiDataset()

def ga_dataloaders(config, ga_seq, ga_fitness, skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  # train_set = GosaiDataset()
  # randomly sample a subset of the train_set as valid_set
  # valid_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))
  # test_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))
  train_set = GaDataset('train', ga_seq, ga_fitness)
  valid_set = GaDataset('val', ga_seq, ga_fitness)
  test_set = GaDataset('test', ga_seq, ga_fitness)

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=not config.data.streaming,
    persistent_workers=True)
  if skip_valid:
    valid_loader = None
    test_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)

  return train_loader, valid_loader, test_loader

def get_dataloaders_gosai(config, skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  # train_set = GosaiDataset()
  # randomly sample a subset of the train_set as valid_set
  # valid_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))
  # test_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))
  train_set = EnhancerDataset('train')
  valid_set = EnhancerDataset('val')
  test_set = EnhancerDataset('test')

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=not config.data.streaming,
    persistent_workers=True)
  if skip_valid:
    valid_loader = None
    test_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)

  return train_loader, valid_loader, test_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py
class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0

