from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule

import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

from Dataset import CardiacDataset
from model import UNet

seq = iaa.Sequential([
    iaa.Affine(scale = (0.75, 1.25),
              rotate = (-45, 45)),
    iaa.ElasticTransformation()
])

train_path = Path("Preprocessed/train/")
val_path = Path("Preprocessed/val")

train_dataset = CardiacDataset(train_path, seq)
val_dataset = CardiacDataset(val_path, None)

batch_size = 8
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size = batch_size,
                                          num_workers = num_workers,
                                          shuffle = True,
                                          )
val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = batch_size,
                                        num_workers = num_workers,
                                        shuffle = False,
                                        )
                                        
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, mask):
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        
        counter = (pred * mask).sum()
        denum = pred.sum() + mask.sum() + 1e-8
        dice = (2 * counter) / denum
        return 1 - dice
        
class AtriumSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UNet()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.loss_fn = DiceLoss()
        
    def forward(self, data):
        return torch.sigmoid(self.model(data))
    
    def training_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)
        
        loss = self.loss_fn(pred, mask)
        
        self.log("Train Dice", loss)
        
        if batch_idx % 50 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Train")
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)
        
        loss = self.loss_fn(pred, mask)
        
        self.log("Val Dice", loss)
        
        if batch_idx % 2 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Val")
            
        return loss
    
    def log_images(self, mri, pred, mask, name):
        
        pred = pred > 0.5
        
        fig, axis = plt.subplots(1, 2)
        
        axis[0].imshow(mri[0][0], cmap = "bone")
        mask_ = np.ma.masked_where(mask[0][0] == 0, pred[0][0])
        axis[0].imshow(mask_, alpha = 0.6)
        
        axis[1].imshow(mri[0][0], cmap = "bone")
        mask_ = np.ma.masked_where(mask[0][0] == 0, pred[0][0])
        axis[1].imshow(mask_, alpha = 0.6)
        
        self.logger.experiment.add_figure(name, fig, self.global_step)
        
    def configure_optimizers(self):
        return [self.optimizer]
         