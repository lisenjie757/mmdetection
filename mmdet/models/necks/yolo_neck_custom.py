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
        p5_up = F.interpolate(p5, scale_factor=2)

        p4 = torch.cat([c4, p5_up], 1)
        p4 = self.convset2(p4)
        p4_up = F.interpolate(p4, scale_factor=2)

        p3 = torch.cat([c3, p4_up], 1)
        p3 = self.convset1(p3)

        p3_down = self.downsample1(p3)
        p4 = torch.cat([p3_down, p4], 1)

        p4_down = self.downsample2(p4)
        p5 = torch.cat([p4_down, p5], 1)

        p5 = self.output(p5)
        
        return tuple([p5])