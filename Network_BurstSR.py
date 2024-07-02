import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.transforms import ToPILImage

from pytorch_lightning import seed_everything

from utils.metrics import PSNR
from utils.postprocessing_functions import SimplePostProcess, BurstSRPostProcess
psnr_fn = PSNR(boundary_ignore=40)

import models
from pwcnet.pwcnet import PWCNet
from utils.metrics import AlignedL1, AlignedL1_loss, AlignedL2_loss, AlignedSSIM_loss, AlignedPSNR, AlignedSSIM, AlignedLPIPS, AlignedLPIPS_loss

seed_everything(13)
post_process = SimplePostProcess(return_np=True)


##############################################################################################
######################################### BurstM #############################################
##############################################################################################

class BurstM(pl.LightningModule):
    def __init__(self):
        super(BurstM, self).__init__()        
        PWCNet_weight_PATH = 'pwcnet/network-default.pth'
        alignment_net = PWCNet(load_pretrained=True, weights_path=PWCNet_weight_PATH)
        for param in alignment_net.parameters():
            param.requires_grad = False
        alignment_net = alignment_net.cuda()
        
        self.aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)
        self.aligned_L1_loss = AlignedL1(alignment_net=alignment_net)

        self.train_loss = nn.L1Loss()
        self.train_loss2 = nn.MSELoss()
        self.valid_psnr = PSNR(boundary_ignore=40)
        
        self.burstm_model = models.BurstM.Neural_Warping().cuda()

    
    def forward(self, burst, step='training', scale=4, target_size=(192,192)):
        
        burst = burst[0]
        burst_ref = burst[0].unsqueeze(0).clone()
        burst_src = burst[1:]
        
        burst_feat, ref, EstLrImg = self.burstm_model(burst_ref, burst_src, scale, target_size)
        
        return burst_feat, ref, EstLrImg
    
    def training_step(self, train_batch, batch_idx):
        x, y, meta_info_burst, meta_info_gt, downsample_factor, target_size, burst_name = train_batch
        pred, ref, EstLrImg = self.forward(x, 'training', downsample_factor.item(), target_size)
        pred = pred.clamp(0.0, 1.0)
        loss = self.aligned_L1_loss(pred, y, x) + self.train_loss2(EstLrImg, ref)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, meta_info_burst, meta_info_gt, downsample_factor, target_size, burst_name = val_batch
        pred, ref, EstLrImg = self.forward(x, 'validation', downsample_factor.item(), target_size)
        pred = pred.clamp(0.0, 1.0)
        
        PSNR = self.aligned_psnr_fn(pred, y, x)

        return PSNR

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        PSNR = torch.stack(outs).mean()
        self.log('val_psnr', PSNR, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):  
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25, eta_min=1e-6)            
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