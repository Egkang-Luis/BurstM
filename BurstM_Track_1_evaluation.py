## Burst Image Restoration and Enhancement
## Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2110.03680

import os
import cv2
import torch
import argparse
import numpy as np
import random
from pytorch_lightning import Trainer, seed_everything
seed_everything(13)
import time
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES']='2'
######################################## Model and Dataset ########################################################
from Network import BurstM
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader
from utils.metrics import PSNR
from utils.postprocessing_functions import SimplePostProcess


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

parser.add_argument('--scale', default='4', type=str, help='Sacle of SR')
parser.add_argument('--input_dir', default='./Zurich-RAW-to-DSLR-Dataset', type=str, help='Directory of inputs')
parser.add_argument('--result_dir', default='./Results/Synthetic/', type=str, help='Directory for results')
parser.add_argument('--result_gt_dir', default='./Results/Synthetic_gt/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained/epoch=294-val_psnr=42.83.ckpt', type=str, help='Path to weights')

args = parser.parse_args()

######################################### Load BIPNet ####################################################

torch_seed(13)
model = BurstM()
model = model.load_from_checkpoint(args.weights, strict=False)
model.cuda()

out_dir = args.result_dir
out_dir_gt = args.result_gt_dir

os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir_gt, exist_ok=True)

dummy_input = torch.rand(1,14,4,48,48, device = 'cuda')
for _ in range(10):
    _ = model(dummy_input)

tt = []
psnrs = []

######################################### Synthetic Burst Validation set #####################################

test_zurich_raw2rgb = ZurichRAW2RGB(root=args.input_dir,  split='test')
test_dataset = SyntheticBurst(test_zurich_raw2rgb, burst_size=14, crop_sz=384, phase='eval', scale_factor=float(args.scale))
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False)

##############################################################################################################    

psnr_fn = PSNR(boundary_ignore=40)
postprocess_fn = SimplePostProcess(return_np=True)

for i, data in tqdm(enumerate(test_loader)):
    x, y, flow_vectors, meta_info, downsample_factor, target_size = data
    x = x.cuda()
    y = y.cuda()
    
    with torch.no_grad():
        tic = time.time()
        output, ref, EstLrImg = model(x, downsample_factor.item(), target_size)
        output = output.clamp(0.0, 1.0)
        toc = time.time()
        tt.append(toc-tic)
        # print('IDX : {}, TIME : {}'.format(i, toc-tic))
    
    psnr = psnr_fn(output, y)
    psnrs.append(psnr.item())
    # ssims.append(ssim.item())
    # lpipss.append(lpips.item())
    
    gt = postprocess_fn.process(y[0].cpu(), meta_info)
    gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
    cv2.imwrite('{}/{}_gt.png'.format(out_dir_gt, i), gt)

    sr_ = postprocess_fn.process(output[0].cpu(), meta_info)
    sr_ = cv2.cvtColor(sr_, cv2.COLOR_RGB2BGR)
    cv2.imwrite('{}/{}_x{}_pred.png'.format(out_dir, i, float(args.scale)), sr_)

    del x
    del output
    del y


print(f'avg PSNR: {np.mean(psnrs):.6f}')
# print(f'avg SSIM: {np.mean(ssims):.6f}')
# print(f'avg LPIPS: {np.mean(lpipss):.6f}')
print(f' avg time: {np.mean(tt):.6f}')
print('BurstM Synthetic x4 result')