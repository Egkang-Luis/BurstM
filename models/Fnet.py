import torch
import torch.nn as nn
import torch.nn.functional as func
import models.utils as utils

class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = func.relu(out)
        out = self.conv2(out)
        out = input + out
        return out


class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = func.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = func.leaky_relu(out, 0.2)
        return out


class FNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, typ):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if typ == "maxpool":
            self.final = lambda x: func.max_pool2d(x, kernel_size=2)
        elif typ == "bilinear":
            self.final = lambda x: func.interpolate(x, scale_factor=2, mode="bilinear")
        else:
            raise Exception('Type does not match any of maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out

class FNetp(nn.Module):
    def __init__(self, in_dim=8):
        super(FNetp, self).__init__()
        self.convPool1 = FNetBlock(in_dim, 32, typ="maxpool") # h/2, w/2
        self.convPool2 = FNetBlock(32, 64, typ="maxpool") # h/4, w/4
        self.convPool3 = FNetBlock(64, 128, typ="maxpool") # h/8, w/8
        self.convleaky1 = ConvLeaky(128, 256) # h/4, w/4
        self.convleaky2 = ConvLeaky(256, 128) # h/2, w/2
        self.convleaky3 = ConvLeaky(128, 64) # h, w
        self.seq = nn.Sequential(self.convPool1, self.convPool2, self.convPool3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, ref, src):
        b, c, h, w = ref.shape
        sizes = (h, w)
        input = torch.cat((ref, src), dim=1)
        
        flow = self.seq(input)
        flow = self.convleaky1(flow)
        flow = func.interpolate(flow, size=(h//4, w//4), mode='bilinear', align_corners=False)
        flow = self.convleaky2(flow)
        flow = func.interpolate(flow, size=(h//2, w//2), mode='bilinear', align_corners=False)
        flow = self.convleaky3(flow)
        flow = func.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
        flow = self.conv1(flow)
        flow = func.leaky_relu(flow, 0.2)
        flow = self.conv2(flow)
        flow = torch.tanh(flow)
        flow = flow.permute(0,2,3,1)

        coord = utils.to_pixel_samples(None, sizes=sizes)
        coord = coord.repeat(b, 1, 1, 1).cuda()
        coord = coord.flip(-1)
        relative_place = coord + flow
        EstLrImg = func.grid_sample(src, relative_place, align_corners=False)
        return flow, ref, EstLrImg