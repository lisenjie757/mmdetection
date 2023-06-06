import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class YOLOK210Neck(nn.Module):

    def __init__(self):
        super(YOLOK210Neck, self).__init__()
        
        def conv_dw(in_channels, out_channels, kernel_size, strides, padding):
                return nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size, strides, padding, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
                    nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), 
                    nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))
        
        # stride = 32
        self.convset3 = nn.Sequential(
            conv_dw(320,96,kernel_size=3,strides=1,padding=1),
        )

        # stride = 16
        self.convset2 = nn.Sequential(
            conv_dw(96+96,96,kernel_size=3,strides=1,padding=1),
        )

        # stride = 8
        self.convset1 = nn.Sequential(
            conv_dw(32+96,96,kernel_size=3,strides=1,padding=1),
        )

        # stride = 8
        self.downsample1 = nn.Sequential(
            conv_dw(96,96,kernel_size=3,strides=2,padding=1),
        )

        # stride = 16
        self.downsample2 = nn.Sequential(
            conv_dw(96+96,96,kernel_size=3,strides=2,padding=1),
        )

        # stride = 32
        self.output = nn.Sequential(
            conv_dw(96+96,96,kernel_size=3,strides=1,padding=1),
        )

    def forward(self, feats):
        c3, c4, c5 = feats

        p5 = self.convset3(c5)
        
        return tuple([p5])

@MODELS.register_module()
class YOLOFastNeck(nn.Module):

    def __init__(self):
        super(YOLOK210Neck, self).__init__()
        
        def conv_dw(in_channels, out_channels, kernel_size, strides, padding):
                return nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size, strides, padding, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
                    nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), 
                    nn.BatchNorm2d(out_channels))
        
        def cbr(in_channels, out_channels, kernel_size, strides, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))
        
        # stride = 32
        self.p5_in = nn.Sequential(
            cbr(384,96,kernel_size=1,strides=1,padding=0),
        )

        # stride = 32
        self.p5_out = nn.Sequential(
            conv_dw(96,96,kernel_size=3,strides=1,padding=1),
            conv_dw(96,96,kernel_size=3,strides=1,padding=1),
        )

        # stride = 16
        self.p4_out = nn.Sequential(
            conv_dw(96+48,144,kernel_size=3,strides=1,padding=1),
            conv_dw(144,144,kernel_size=3,strides=1,padding=1),

        )

        # output fusion
        self.output = nn.Sequential(
            conv_dw(144+96,96,kernel_size=3,strides=1,padding=1),
        )

    def forward(self, feats):
        c3, c4, c5 = feats

        # SPP
        spp1 = c5
        spp2 = F.max_pool2d(c5, kernel_size=3, stride=1, padding=1)
        spp3 = F.max_pool2d(c5, kernel_size=5, stride=1, padding=2)
        spp4 = F.max_pool2d(c5, kernel_size=9, stride=1, padding=4)
        p5 = torch.cat([spp1, spp2, spp3, spp4], dim=1)

        p5_in = self.p5_in(p5)
        p5_out = self.p5_out(p5_in)

        p5_up = F.interpolate(p5_in, scale_factor=2)
        p4 = torch.cat([p5_up, c4], dim=1)
        p4_out = self.p4_out(p4)

        output = torch.cat([p4_out, p5_out], dim=1)

        
        return tuple([p5])