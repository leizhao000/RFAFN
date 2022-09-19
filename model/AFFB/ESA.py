import torch
import torch.nn as nn
from model import block as B
import torch.nn.functional as F


class SCAB(nn.Module):
    def __init__(self,in_channels):
        super(SCAB,self).__init__()
        self.conv1=B.conv_layer(in_channels,in_channels//2,3)
        self.conv2=B.conv_layer(in_channels//2,in_channels//2,3,groups=4)
        self.conv3=B.conv_layer(in_channels // 2, in_channels // 2, 3)
        self.conv4 = B.conv_layer(in_channels // 2, in_channels // 2, 3,groups=4)
        self.conv5=B.conv_layer(in_channels,in_channels,1)
        self.cam = B.CALayer(in_channels)
        self.sigmoid=nn.Sigmoid()
        self.act=B.activation('gelu')
    def forward(self,x):
        out1=self.act(self.conv1(x))
        out_1=self.conv2(out1)
        out2=self.conv3(out1)
        out3=self.sigmoid(out_1)*out2
        out_2=self.conv4(out3)
        out=torch.cat([out_1,out_2],dim=1)
        out=self.act(self.conv5(self.cam(out)))
        return out+x

class HFFM(nn.Module):
    def __init__(self, in_channels):
        super(HFFM, self).__init__()
        self.B1 = SCAB(in_channels)
        self.B2 = SCAB(in_channels)
        self.B3 = SCAB(in_channels)

        self.c1=B.conv_layer(in_channels*2,in_channels,1)
        self.c2 = B.conv_layer(in_channels * 2, in_channels, 1)
        self.sam1 =B.ESA(in_channels)
        self.sam2 = B.ESA(in_channels)
        self.act=B.activation('gelu')

    def forward(self, input):
        out1 = self.B1(input)
        c1 = self.sam1(self.c1(torch.cat([input,out1],dim=1)))
        out2 = self.B2(c1)
        c2 = self.sam2(self.c2(torch.cat([out1,out2],dim=1)))
        out3 = self.B3(c2)
        return out3

class LFIFN(nn.Module):
    def __init__(self, in_channels=3 ,out_channels=64, scale=2):
        super(LFIFN, self).__init__()
        self.fea_conv = B.conv_layer(in_channels, out_channels, kernel_size=3)
        self.act = B.activation('gelu')
        self.scale=scale
        self.HFFM1=HFFM(out_channels)
        self.HFFM2=HFFM(out_channels)
        self.HFFM3=HFFM(out_channels)
        self.HFFM4=HFFM(out_channels)
        self.c1 = B.conv_layer(out_channels * 2, out_channels, kernel_size=1)  # 特征聚合
        self.c2 = B.conv_layer(out_channels * 2, out_channels, kernel_size=1)  # 特征聚合
        self.c3 = B.conv_layer(out_channels * 2, out_channels, kernel_size=1)  # 特征聚合
        self.c4= B.conv_layer(out_channels * 2, out_channels, kernel_size=1)  # 特征聚合
        self.c=B.conv_layer(out_channels ,out_channels, kernel_size=3)  # 特征聚合
        self.upsampler = B.pixelshuffle_block(out_channels, in_channels, upscale_factor=self.scale)

    def forward(self,input):
        out_fea=self.fea_conv(input)
        out_upscale=F.interpolate(input, scale_factor=self.scale, mode='bilinear',align_corners=False)
        B1 = self.HFFM1(out_fea)
        c1=self.act(self.c1(torch.cat([out_fea,B1],dim=1)))
        B2 = self.HFFM2(B1)
        c2 = self.act(self.c2(torch.cat([c1, B2], dim=1)))
        B3 = self.HFFM3(B2)
        c3 = self.act(self.c3(torch.cat([c2, B3], dim=1)))
        B4 = self.HFFM4(B3)
        out_B = self.act(self.c4(torch.cat([c3,B4], dim=1)) ) # 特征聚合
        out_lr =out_fea+self.c(out_B)
        output =self.upsampler(out_lr)+out_upscale

        return output