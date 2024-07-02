import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from utils.metrics import PSNR
from utils.postprocessing_functions import SimplePostProcess
psnr_fn = PSNR(boundary_ignore=40)

import models

seed_everything(13)
post_process = SimplePostProcess(return_np=True)


class BurstM(pl.LightningModule):
    def __init__(self):
        super(BurstM, self).__init__()        
        
        self.train_loss = nn.L1Loss()
        self.train_loss2 = nn.MSELoss()
        self.valid_psnr = PSNR(boundary_ignore=40)

        self.burstm_model = models.BurstM.Neural_Warping().cuda()
        
    
    def forward(self, burst, scale=4, target_size=(192,192)):
        
        burst = burst[0]
        burst_ref = burst[0].unsqueeze(0).clone()
        burst_src = burst[1:]
        
        burst_feat, ref, EstLrImg = self.burstm_model(burst_ref, burst_src, scale, target_size)
        
        return burst_feat, ref, EstLrImg
    
    def training_step(self, train_batch, batch_idx):
        x, y, flow_vectors, meta_info, downsample_factor, target_size = train_batch
        pred, ref, EstLrImg = self.forward(x, downsample_factor.item(), target_size)
        pred = pred.clamp(0.0, 1.0)
        loss = self.train_loss(pred, y) + self.train_loss2(EstLrImg, ref)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, flow_vectors, meta_info, downsample_factor, target_size = val_batch
        pred, ref, EstLrImg = self.forward(x, downsample_factor.item(), target_size)
        pred = pred.clamp(0.0, 1.0)
        PSNR = self.valid_psnr(pred, y)
        
        return PSNR


    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        PSNR = torch.stack(outs).mean()
        self.log('val_psnr', PSNR, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):  
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6)            
        # return [optimizer], [lr_scheduler]
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',  # or 'step' if you want to update per batch
                'frequency': 1
            }
        }


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)