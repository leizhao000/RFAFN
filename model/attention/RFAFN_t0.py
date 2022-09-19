import torch
import torch.nn as nn
from model import block as B
import torch.nn.functional as F

class AFF(nn.Module):
    def __init__(self,channel,reduction=16):
        super(AFF, self).__init__()
        self.att=B.ESAM1(channel,reduction=reduction)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Conv2d(channel//reduction, channel, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        self.c=B.conv_layer(channel*2, channel, 1)

    def forward(self, x1,x2):
        feats = torch.cat([x1,x2], dim=1)
        feats1 = feats.view(feats.shape[0], 2, -1, feats.shape[2], feats.shape[3])
        feats2 = torch.sum(feats1, dim=1)
        attention=self.att(feats2)
        attention_vectors = [fc(attention) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(feats.shape[0], 2, -1, feats.shape[2], feats.shape[3])
        attention_vectors = self.softmax(attention_vectors)*feats1
        for i in range(2):
            out_cam = attention_vectors[:, i, :, :]
            if i == 0:
                out = out_cam
            else:
                out = torch.cat((out, out_cam), 1)
        out=self.c(out)
        return out

class SCAB(nn.Module):
    def __init__(self,in_channels):
        super(SCAB,self).__init__()
        self.conv1=B.conv_layer(in_channels,in_channels//2,3)
        self.conv2=B.conv_layer(in_channels//2,in_channels//2,3,groups=4)
        self.conv3=B.conv_layer(in_channels // 2, in_channels // 2, 3)
        self.conv4 = B.conv_layer(in_channels // 2, in_channels // 2, 3,groups=4)
        self.conv5=B.conv_layer(in_channels,in_channels,1)
        self.cam=B.CALayer(in_channels)
        self.sigmoid=nn.Sigmoid()
        self.act=B.activation('lrelu')
    def forward(self,x):
        out1=self.act(self.conv1(x))
        out_1=self.conv2(out1)
        out2=self.conv3(out1)
        out3=self.sigmoid(out_1)*out2
        out_2=self.conv4(out3)
        out=torch.cat([out_1,out_2],dim=1)
        out=self.act(self.conv5(self.cam(out)))
        return out

class SCAB1(nn.Module):
    def __init__(self, in_channels):
        super(SCAB1, self).__init__()
        self.B1 =B.conv_layer(in_channels,in_channels//2,3)
        self.B2 = B.conv_layer(in_channels//2, in_channels, 3)
        self.act = B.activation('lrelu')
    def forward(self, input):
        out1 = self.act(self.B1(input))
        out2 = self.act(self.B2(out1))
        return out2+input

class HFFM(nn.Module):
    def __init__(self, in_channels):
        super(HFFM, self).__init__()
        self.B1 = SCAB(in_channels)
        self.B2 = SCAB(in_channels)
        self.B3 = SCAB(in_channels)

        self.c1 = AFF(in_channels)
        self.c2 = B.conv_layer(in_channels*2,in_channels,1)
        self.act = B.activation('lrelu')

    def forward(self, input):
        out1 = self.B1(input)
        c1 = self.act(self.c1(input,out1))
        out2 = self.B2(c1)
        c2 = self.act(self.c2(torch.cat([out1,out2],dim=1)))
        out3 = self.B3(c2)
        return out3+input

class LFIFN(nn.Module):
    def __init__(self, in_channels=3 ,out_channels=64, scale=4):
        super(LFIFN, self).__init__()
        self.fea_conv = B.conv_layer(in_channels, out_channels, kernel_size=3)
        self.act = B.activation('lrelu')
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
        #out_upscale=F.interpolate(input, scale_factor=self.scale, mode='bilinear',align_corners=False)
        B1 = self.HFFM1(out_fea)
        c1=self.act(self.c1(torch.cat([out_fea,B1],dim=1)))
        B2 = self.HFFM2(B1)
        c2 = self.act(self.c2(torch.cat([c1, B2], dim=1)))
        B3 = self.HFFM3(B2)
        c3 = self.act(self.c3(torch.cat([c2, B3], dim=1)))
        B4 = self.HFFM4(B3)
        out_B = self.act(self.c4(torch.cat([c3,B4], dim=1)) ) # 特征聚合
        out_lr =self.c(out_B)
        output =self.upsampler(out_lr)

        return output
