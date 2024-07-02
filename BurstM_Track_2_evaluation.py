import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F

######################################## Model and Dataset ########################################################
from Network_BurstSR import BurstM
###################################################################################################################

###################################################################################################################
from torch.utils.data.dataloader import DataLoader
from datasets.burstsr_dataset import BurstSRDataset

from utils.data_format_utils import convert_dict
from utils.postprocessing_functions import BurstSRPostProcess

from utils.metrics import AlignedPSNR, AlignedSSIM, AlignedLPIPS
from pwcnet.pwcnet import PWCNet

import data_processing.camera_pipeline as rgb2raw
from data_processing.camera_pipeline import *
import time
##################################################################################################################


parser = argparse.ArgumentParser(description='Real burst super-resolution using BurstM')

parser.add_argument('--scale', default='4', type=str, help='Sacle of SR')
parser.add_argument('--input_dir', default='./BurstSR/', type=str, help='Directory for inputs')
parser.add_argument('--result_dir', default='./Results/Real/', type=str, help='Directory for results')
parser.add_argument('--result_gt_dir', default='./Results/Real_gt/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained/epoch=16-val_psnr=49.12.ckpt', type=str, help='Path to pre-trained weights')

args = parser.parse_args()

##################################################################################################################


class BurstSR_Test_Network():

    def __init__(self, args):
        super().__init__()
        
        PWCNet_weight_PATH = 'pwcnet/network-default.pth'
        alignment_net = PWCNet(load_pretrained=True, weights_path=PWCNet_weight_PATH)
        alignment_net = alignment_net.cuda()
        
        self.aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net)
        self.aligned_ssim_fn = AlignedSSIM(alignment_net=alignment_net)
        self.aligned_lpips_fn = AlignedLPIPS(alignment_net=alignment_net)
        
    def test(self):               
        # Postprocessing function to obtain sRGB images
        postprocess_fn = BurstSRPostProcess(return_np=True)

        result_dir = args.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)     
        gt_dir = args.result_gt_dir
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir, exist_ok=True)     
        
        model = BurstM()                
        model = model.load_from_checkpoint(args.weights, strict=False)
        model.eval()
        model.cuda() 
        
        ################## DATA Loaders ########################################
        
        test_dataset = BurstSRDataset(root=args.input_dir,  split='val', burst_size=14, crop_sz=80, random_flip=False)
        test_data_loader = DataLoader(test_dataset, batch_size=1)
        
        dummy_input = torch.rand(1,14,4,80,80, device = 'cuda')
        for _ in range(10):
            _ = model(dummy_input)
        
        tt = []
        PSNR = []
        LPIPS = []
        SSIM = []
        
        if args.scale == '4':
            for i, data in enumerate(test_data_loader):
                burst, labels, meta_info_burst, meta_info_gt, downsample_factor, target_size, burst_name = data
                
                target_size[0] = int(burst.shape[-1]*float(args.scale))
                target_size[1] = int(burst.shape[-2]*float(args.scale))
                
                meta_info_burst = convert_dict(meta_info_burst, burst.shape[0])
                
                burst_rgb = rgb2raw.demosaic(burst[0])            
                burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
                burst_rgb = F.interpolate(burst_rgb, scale_factor=4, mode='bilinear', align_corners=True)
                
                burst = burst.cuda()
                labels = labels.cuda()
                output = labels*0
                with torch.no_grad():
                    tic = time.time()
                    output, ref, EstLrImg = model(burst, 'validation', downsample_factor.item(), target_size)
                    toc = time.time()
                    tt.append(toc-tic)
                    # print('IDX : {}, TIME : {}'.format(i, toc-tic))
                    output = output
                    output = output.clamp(0.0, 1.0)
                
                PSNR_temp = self.aligned_psnr_fn(output, labels, burst).cpu().numpy()            
                PSNR.append(PSNR_temp)
                
                LPIPS_temp = self.aligned_lpips_fn(output, labels, burst).cpu().detach().numpy()
                LPIPS.append(LPIPS_temp)
                
                SSIM_temp = self.aligned_ssim_fn(output, labels, burst).cpu().numpy()
                SSIM.append(SSIM_temp)
                
                print('Evaluation Measures for Burst {:d} ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f} \n'.format(i, PSNR_temp, SSIM_temp, LPIPS_temp))
                
                burst = burst.cpu()
                output = output.cpu()
                labels = labels.cpu()
                
                meta_info_gt = convert_dict(meta_info_gt, labels.shape[0])

                # Apply simple post-processing to obtain RGB images
                input_burst = postprocess_fn.process(burst_rgb[0], meta_info_burst[0])
                output = postprocess_fn.process(output[0], meta_info_gt[0])
                labels = postprocess_fn.process(labels[0], meta_info_gt[0])
                
                input_burst = cv2.cvtColor(input_burst, cv2.COLOR_RGB2BGR)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

                #output = np.concatenate((input_burst, output, labels), axis=1)
                
                cv2.imwrite('{}/{}'.format(result_dir, burst_name[0] +'_x'+ args.scale + '_pred.png'), output)
                cv2.imwrite('{}/{}'.format(gt_dir, burst_name[0] +'_gt.png'), labels)
            
            Average_PSNR = sum(PSNR)/len(PSNR)
            Average_SSIM = sum(SSIM)/len(SSIM)
            Average_LPIPS = sum(LPIPS)/len(LPIPS)
            Average_time = np.mean(tt)
            average_eval_par = '\nAverage Evaluation Measures ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f} and Inference time is {:0.3f}\n'.format(Average_PSNR, Average_SSIM, Average_LPIPS, Average_time)
                        
            print(average_eval_par)
            
        else:
            for i, data in enumerate(test_data_loader):
                burst, labels, meta_info_burst, meta_info_gt, downsample_factor, target_size, burst_name = data
                
                target_size[0] = int(burst.shape[-1]*float(args.scale))
                target_size[1] = int(burst.shape[-2]*float(args.scale))
                
                meta_info_burst = convert_dict(meta_info_burst, burst.shape[0])
                
                burst_rgb = rgb2raw.demosaic(burst[0])            
                burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
                burst_rgb = F.interpolate(burst_rgb, scale_factor=4, mode='bilinear', align_corners=True)
                
                burst = burst.cuda()
                labels = labels.cuda()
                output = labels*0
                with torch.no_grad():
                    tic = time.time()
                    output, ref, EstLrImg = model(burst, 'validation', downsample_factor.item(), target_size)
                    toc = time.time()
                    tt.append(toc-tic)
                    # print('IDX : {}, TIME : {}'.format(i, toc-tic))
                    
                    output = output
                    output = output.clamp(0.0, 1.0)
                
                print('Burst {:d} ::: Can not evaluation by PSNR, SSIM, LPIPS\n'.format(i))
                
                burst = burst.cpu()
                output = output.cpu()
                labels = labels.cpu()
                
                meta_info_gt = convert_dict(meta_info_gt, labels.shape[0])

                # Apply simple post-processing to obtain RGB images
                input_burst = postprocess_fn.process(burst_rgb[0], meta_info_burst[0])
                output = postprocess_fn.process(output[0], meta_info_gt[0])
                labels = postprocess_fn.process(labels[0], meta_info_gt[0])
                
                input_burst = cv2.cvtColor(input_burst, cv2.COLOR_RGB2BGR)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

                #output = np.concatenate((input_burst, output, labels), axis=1)
                
                cv2.imwrite('{}/{}'.format(result_dir, burst_name[0] +'_x'+ args.scale + '_pred.png'), output)
                cv2.imwrite('{}/{}'.format(gt_dir, burst_name[0] +'_gt.png'), labels)
            print(f' Average time: {np.mean(tt):.6f}')

BurstSR_Test_Network(args).test()


