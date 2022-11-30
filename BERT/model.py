from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import transformers
import pytorch_lightning as pl
from torchmetrics import AUC, Precision, Recall
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch


BERT_MODEL_NAME = "bert-base-uncased"
scheduler_factor = 0.5
scheduler_patience = 2

class BertClassifier(pl.LightningModule):

    def __init__(self, n_classes: int, n_training_steps:int,learning_rate:float,weight_decay:float):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.BCELoss()
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(output.pooler_output)
        output = self.classifier(output)
        output = torch.sigmoid(output)
        output = output.view(-1)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.float())
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        ac = Accuracy().to(device='cuda:0')
        acc = ac(outputs,labels).to(device='cuda:0')
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        ac = Accuracy().to(device='cuda:0')

        acc = ac(outputs,labels).to(device='cuda:0')
        P = Precision().to(device='cuda:0')
        re = Recall().to(device='cuda:0')
        precision =  P(outputs,labels)
        recall = re(outputs,labels)
        f1 = F1Score().to(device='cuda:0')
        f1_ = f1(outputs,labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_f1", f1_, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_precision", precision, prog_bar=True, logger=True)
        self.log("val_recall", recall, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=scheduler_factor, patience=scheduler_patience)
        return dict(
        optimizer=optimizer,
        lr_scheduler=dict(
        scheduler=scheduler,
        interval='epoch',
        monitor="val_loss",
        )
    )

def Model(n_classes=2,n_training_steps=100,learning_rate=2e-5,weight_decay=0):

    model = BertClassifier(n_classes=n_classes,n_training_steps=n_training_steps,learning_rate=learning_rate,weight_decay=weight_decay)

    return model
