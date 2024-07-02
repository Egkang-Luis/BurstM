import torch
import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import models.utils as utils


class Neural_Warping(nn.Module):
    def __init__(self, encoder_spec='edsr-baseline', blender_spec='edsr-baseline', hidden_dim=128):
        super().__init__()
        encoder_spec = {'name':'edsr-baseline',
                        'args': {'no_upsampling':True, 'input_channel':4, 'n_resblocks':16, 'n_feats': 128, 'res_scale':1, 'conv': 'default_conv'}}
        blender_spec = {'name': 'edsr-baseline',
                        'args': {'no_upsampling':True, 'input_channel':hidden_dim*14, 'n_resblocks':16, 'n_feats': 128, 'n_colors': 512, 'res_scale':1,'conv': 'default_conv'}}

        self.encoder = models.make(encoder_spec)
        self.blender = models.make(blender_spec)
        
        self.up = nn.PixelShuffle(2)

        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Conv2d(2, hidden_dim//2, 1, 1, 0, bias=False)

        self.stem = nn.Conv2d(1, 1, 1, 1, 0)

        # offset generation
        self.fnetp = models.Fnet.FNetp()
        self.imnet = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(256, 12, 1, 1, 0),
        )
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.skipup1 = nn.Conv2d(1, 128, 3, 1, 1, bias=True)
        self.skipup2 = nn.Conv2d(128, 3 * 4, 3, 1, 1, bias=True)


    def gen_feat(self, inp):
        feat_coord = utils.make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])
        
        feat = self.encoder(inp)
        coeff = self.coef(feat)
        freqq = self.freq(feat)

        return feat, feat_coord, coeff, freqq
    

    def LTEW(self, img, feat, feat_coord, freq, coef, coord):
        
        w_coef = F.grid_sample(coef, coord, mode='nearest', align_corners=False)
        w_freq = F.grid_sample(freq, coord, mode='nearest', align_corners=False).permute(0, 2, 3, 1)
        w_coord = F.grid_sample(feat_coord, coord, mode='nearest', align_corners=False).permute(0, 2, 3, 1)
        
        coord = coord.flip(-1)
        rel_coord = coord - w_coord
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]
        
        rel_cell = torch.ones_like(coord)
        rel_cell[..., 0] = rel_cell[..., 0] / feat.shape[-2]
        rel_cell[..., 1] = rel_cell[..., 1] / feat.shape[-1]
        rel_cell = rel_cell.permute(0,3,1,2).contiguous()

        w_freq = torch.stack(torch.split(w_freq, 2, dim=-1), dim=-1)

        w_freq = torch.mul(w_freq, rel_coord.unsqueeze(-1))
        w_freq = torch.sum(w_freq, dim=-2).permute(0, 3, 1, 2)
        w_freq = w_freq + self.phase(rel_cell)

        w_freq = torch.cat((torch.cos(np.pi*w_freq), torch.sin(np.pi*w_freq)), dim=1)

        spatial_signal = torch.mul(w_coef, w_freq)

        return spatial_signal


    def nw_module(self, tgt, tgt_grid, src, src_grid):
        
        sc = F.grid_sample(self.up(tgt), tgt_grid, mode='bilinear', align_corners=False)
        sc = self.skipup2(self.lrelu(self.skipup1(sc)))

        burst_inp = torch.cat([tgt, src], dim=0)
        burst_grid = torch.cat([tgt_grid, src_grid], dim=0)
        fea_burst, fea_burst_grid, burst_coef, burst_freq = self.gen_feat(burst_inp)
        spatial_signal = self.LTEW(
            burst_inp, fea_burst, fea_burst_grid,
            burst_freq, burst_coef, burst_grid)
        
        b, c, h, w = spatial_signal.shape
        spatial_signal= spatial_signal.reshape(1, b*c, h, w)
        spatial_signal = self.blender(spatial_signal)
        spatial_signal = self.imnet(spatial_signal)
        spatial_signal = spatial_signal + sc
        spatial_signal = self.up(spatial_signal)        

        return spatial_signal

    def offset_gen(self, tgt, src):
        tgt_lr = tgt.clone()
        tgt_lr = torch.repeat_interleave(tgt_lr, src.shape[0], dim=0)
        src_lr = src.clone()
        offset, ref, EstLrImg = self.fnetp(tgt_lr, src_lr)

        return offset, ref, EstLrImg



    def forward(self, tgt, src, scale, target_size):
        b, c, h, w = tgt.shape
        sizes = (h, w)
        
        offset_coord, ref, EstLrImg = self.offset_gen(tgt, src)

        coord = utils.to_pixel_samples(None, sizes=sizes).cuda()

        coord1 = coord.clone()
        coord1 = coord1.repeat(b, 1, 1, 1)
        src_grid = utils.gridy2gridx_flow(
            coord1.contiguous(), *target_size, offset_coord, tgt_symbol=False)
        
        coord2 = coord.clone()
        coord2 = coord2.unsqueeze(0)
        tgt_grid = utils.gridy2gridx_flow(
            coord2.contiguous(), *target_size, 0, tgt_symbol=True)

        aligned_img = self.nw_module(
            tgt, tgt_grid, 
            src, src_grid
        )
        
        return aligned_img, ref, EstLrImg