import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning import _logger as log
from pytorch_lightning import LightningDataModule
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info

def SatResNet50(num_classes, pretrained):

    model = models.resnet50(pretrained=pretrained)
    for name, param in model.named_parameters():
        if("bn" not in name):
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2), 
        nn.Linear(512, num_classes))    
    return model

class LitResnet(pl.LightningModule):
    def __init__(self, num_classes, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = SatResNet50(num_classes, pretrained=True)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = 45000 // batch_size
        scheduler_dict = {
            'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}