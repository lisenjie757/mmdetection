import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class MobileNetV1(nn.Module):

    def __init__(self):
        super(MobileNetV1, self).__init__()
    
        def conv_dw(in_channels, out_channels, kernel_size, strides, padding):
                return nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size, strides, padding, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
                    nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), 
                    nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))
        
        # stride = 2
        self.layer1 = nn.Sequential(
            conv_dw(3,4,kernel_size=3,strides=1,padding=1),
            conv_dw(4,8,kernel_size=3,strides=2,padding=1),
            conv_dw(8,8,kernel_size=3,strides=1,padding=1),
        )
        # stride = 4
        self.layer2 = nn.Sequential(
            conv_dw(8,16,kernel_size=3,strides=2,padding=1),
            conv_dw(16,16,kernel_size=3,strides=1,padding=1),
        )
        # stride = 8
        self.layer3 = nn.Sequential(
            conv_dw(16,32,kernel_size=3,strides=2,padding=1),
            conv_dw(32,32,kernel_size=3,strides=1,padding=1),
        )
        # stride = 16
        self.layer4 = nn.Sequential(
            conv_dw(32,96,kernel_size=3,strides=2,padding=1),
            conv_dw(96,96,kernel_size=3,strides=1,padding=1),
        )
        # stride = 32
        self.layer5 = nn.Sequential(
            conv_dw(96,320,kernel_size=3,strides=2,padding=1),
            conv_dw(320,320,kernel_size=3,strides=1,padding=1),
        )

    def forward(self, x):  # should return a tuple
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        return tuple([c3,c4,c5])