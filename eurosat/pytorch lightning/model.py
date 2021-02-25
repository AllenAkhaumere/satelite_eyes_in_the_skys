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
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info