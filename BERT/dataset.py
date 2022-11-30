from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
import torch.utils.data as data
import torch
import pandas as pd
import model
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
BATCH_SIZE=10 #best 10a
N_EPOCHS=50
MAX_TOKEN_LEN = 200 #best 200


class CSDataset(Dataset):

  def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: tokenizer,
    max_token_len: int = MAX_TOKEN_LEN
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]
    text = data_row.sentences
    labels = data_row['category_index']

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      text=text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.tensor(labels)
    )


class CSdatamodule(pl.LightningDataModule):

  def __init__(self, train_df, valid_df, test_df, tokenizer, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_LEN):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.valid_df = valid_df

    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def setup(self, stage=None):
      self.train_dataset = CSDataset(
        self.train_df,
        self.tokenizer,
        self.max_token_len
      )

      self.test_dataset = CSDataset(
        self.test_df,
        self.tokenizer,
        self.max_token_len
      )

      self.valid_dataset = CSDataset(
        self.valid_df,
        self.tokenizer,
        self.max_token_len
      )



  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=0
    )

  def val_dataloader(self):
    return DataLoader(
      self.valid_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )
