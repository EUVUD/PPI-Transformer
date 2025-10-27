import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model.modules.ppiTransformer import ppiTransformer

class ppiPredictor(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = ppiTransformer()
        self.criterion = nn.BCELoss()
        self.lr = lr

    def forward(self, emb1, emb2):
        return self.model(emb1, emb2)

    def training_step(self, batch, batch_idx):
        emb1, emb2, label = batch
        preds = self(emb1, emb2)
        loss = self.criterion(preds.view(-1), label.float().view(-1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        emb1, emb2, label = batch
        preds = self(emb1, emb2)
        loss = self.criterion(preds.view(-1), label.float().view(-1))
        acc = ((preds.view(-1) > 0.5) == label.view(-1)).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
