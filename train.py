from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
import torch.utils.data as data
import torch
import argparse
from pathlib import Path
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
from pytorch_lightning.loggers import CSVLogger

import dataset as dt

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
BATCH_SIZE=10 #best 10
N_EPOCHS=50
MAX_TOKEN_LEN = 200 #best 200

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=Path, help='Path to training dataset')
    parser.add_argument('--batch_size', default=10, type=int, help='Maximum size of batches')
    parser.add_argument('--epochs', default=50, type=int, help='Maximum size of batches')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate of training session')
    parser.add_argument('--l1', default=0.0, type=float, help='Strength of L1 regularization')
    parser.add_argument('--weight_decay', default=0, type=float, help='Strength of weight decay regularization')
    parser.add_argument('--patience', default=20, type=int, help='Patience for early stopping')
    args = parser.parse_args()
    return args



if __name__ == '__main__':



    args = parseArguments()

    df = pd.read_csv(args.train)


    seed_everything(42, workers=True)

    for i in range(10):
        print(df['category_index'].sum())
        # Random split

        train_set_size = int(len(df) * 0.8)
        valid_set_size = len(df) - train_set_size
        train_set, valid_set = train_test_split(df, test_size=0.2)

        valid_set.to_csv("validSimple_"+str(i)+".csv")

        print(len(train_set), len(valid_set))

        data_module = dt.CSdatamodule(
            train_set,
            valid_set,
            valid_set,
            tokenizer,
            batch_size=args.batch_size,
            max_token_len=MAX_TOKEN_LEN)


        steps_per_epoch=len(train_set) // args.batch_size
        total_training_steps = steps_per_epoch * args.epochs
        modelBert = model.Model(
            n_classes=1,
            n_training_steps=total_training_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=args.patience)
        checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}-{val_f1:.2f}-{val_precision:.2f}-{val_recall:.2f}')
        trainer = pl.Trainer(
            enable_checkpointing=True,
            callbacks=[early_stopping_callback,checkpoint_callback],
            max_epochs=args.epochs,
            num_sanity_val_steps=0,
            log_every_n_steps=10,
            deterministic=True,
            gpus=1,
        )

        trainer.fit(modelBert, data_module)


        del modelBert,trainer,early_stopping_callback,data_module
        torch.cuda.empty_cache()


