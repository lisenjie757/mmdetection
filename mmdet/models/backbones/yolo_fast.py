import torch.nn as nn

from mmdet.registry import MODELS

class ResidualBlock(nn.Module):
    def __init__(self, inout_channels, mid_channels):
        super(ResidualBlock, self).__init__()

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
        
        self.conv = nn.Sequential(
            cbr(inout_channels, mid_channels, kernel_size=1,strides=1,padding=0),
            conv_dw(mid_channels, inout_channels, kernel_size=3,strides=1,padding=1),
        )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out


@MODELS.register_module()
class YOLOFast(nn.Module):

    def __init__(self):
        super(YOLOFast, self).__init__()
    
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
             
        
        # stride = 2
        self.layer1 = nn.Sequential(
            cbr(3,16,kernel_size=3,strides=2,padding=1),
            cbr(16,16,kernel_size=1,strides=1,padding=0),
            conv_dw(16,8,kernel_size=3,strides=1,padding=1),
            ResidualBlock(8,16),
            cbr(8,48,kernel_size=1,strides=1,padding=0),
        )
        # stride = 4
        self.layer2 = nn.Sequential(
            conv_dw(48,16,kernel_size=3,strides=2,padding=1),
            ResidualBlock(16,64),
            ResidualBlock(16,64),
            cbr(16,64,kernel_size=1,strides=1,padding=0),
        )
        # stride = 8
        self.layer3 = nn.Sequential(
            conv_dw(64,16,kernel_size=3,strides=2,padding=1),
            ResidualBlock(16,96),
            ResidualBlock(16,96),
            cbr(16,96,kernel_size=1,strides=1,padding=0),
            conv_dw(96,32,kernel_size=3,strides=1,padding=1),
            ResidualBlock(32,192),
            ResidualBlock(32,192),
            ResidualBlock(32,192),
            ResidualBlock(32,192),
            cbr(32,192,kernel_size=1,strides=1,padding=0),
        )
        # stride = 16
        self.layer4 = nn.Sequential(
            conv_dw(192,48,kernel_size=3,strides=2,padding=1),
            ResidualBlock(48,272),
            ResidualBlock(48,272),
            ResidualBlock(48,272),
            ResidualBlock(48,272),
            cbr(48,272,kernel_size=1,strides=1,padding=0),
        )
        # stride = 32
        self.layer5 = nn.Sequential(
            conv_dw(272,96,kernel_size=3,strides=2,padding=1),
            ResidualBlock(96,448),
            ResidualBlock(96,448),
            ResidualBlock(96,448),
            ResidualBlock(96,448),
            ResidualBlock(96,448),
        )

    def forward(self, x):  # should return a tuple
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        return tuple([c3,c4,c5])