import os
import argparse
from setproctitle import *

pjt = 'Reproduce_to_release'
setproctitle(pjt)

######################################## Pytorch lightning ########################################################

import torch
import random
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
seed_everything(13)

######################################## Model and Dataset ########################################################

from Network import BurstM
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader

##################################################################################################################

def torch_seed(random_seed=13):

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

parser = argparse.ArgumentParser(description='Synthetic burst super-resolution using BurstM')

parser.add_argument('--log_dir', default="./Results/SyntheticBurst/tensorboard", type=str, help='Directory of logs(Tensorboard)')
parser.add_argument('--image_dir', default="./Zurich-RAW-to-DSLR-Dataset", type=str, help='Directory of inputs')
parser.add_argument('--model_dir', default="./Results/SyntheticBurst/saved_model", type=str, help='Directory of model')
parser.add_argument('--result_dir', default="./Results/SyntheticBurst/result", type=str, help='Directory of results')
parser.add_argument('--burst_size', default="14", type=int, help='Number of Burst short')
args = parser.parse_args()

######################################### Data loader ######################################################

def load_data(image_dir, burst_size):

    train_zurich_raw2rgb = ZurichRAW2RGB(root=image_dir,  split='train')
    train_dataset = SyntheticBurst(train_zurich_raw2rgb, burst_size=burst_size, crop_sz=384, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=6, pin_memory=True)

    test_zurich_raw2rgb = ZurichRAW2RGB(root=image_dir,  split='test')
    test_dataset = SyntheticBurst(test_zurich_raw2rgb, burst_size=burst_size, crop_sz=384, phase='test', scale_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=6, pin_memory=True)

    return train_loader, test_loader

######################################### Load BurstM ####################################################

torch_seed(13)
model = BurstM()
model.cuda()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir, exist_ok=True) 

######################################### Training #######################################################

train_loader, test_loader = load_data(args.image_dir, args.burst_size)

checkpoint_callback = ModelCheckpoint(
    monitor='val_psnr',
    dirpath=args.model_dir,
    filename='{epoch:02d}-{val_psnr:.2f}',
    save_top_k=3,
    save_last=True,
    mode='max',
)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir, version=0)

trainer = Trainer(gpus=4,
                    auto_select_gpus=True,
                    accelerator='ddp',
                    max_epochs=300,
                    precision=16,
                    gradient_clip_val=0.01,
                    callbacks=[checkpoint_callback],
                    val_check_interval=0.25,
                    progress_bar_refresh_rate=1,
                    profiler="simple",
                    logger=tb_logger)

trainer.fit(model, train_loader, test_loader)
