from model.model1 import common
import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F

def make_model(args, parent=False):
    return IDN(args)

# DBlocks
class Enhancement_unit(nn.Module):
    def __init__(self, nFeat, nDiff, nFeat_slice):
        super(Enhancement_unit, self).__init__()

        self.D3 = nFeat
        self.d = nDiff
        self.s = nFeat_slice

        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))       
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-2*nDiff, kernel_size=3, padding=1, bias=True, groups=4))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-2*nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        self.conv_block0 = nn.Sequential(*block_0)

        block_1 = [] 
        block_1.append(nn.Conv2d(nFeat-nFeat//4, nFeat, kernel_size=3, padding=1, bias=True))        
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True, groups=4))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat+nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        self.conv_block1 = nn.Sequential(*block_1)
        self.compress = nn.Conv2d(nFeat+nDiff, nFeat, kernel_size=1, padding=0, bias=True)
    def forward(self, x):

        x_feature_shot = self.conv_block0(x)
        feature = x_feature_shot[:,0:(self.D3-self.D3//self.s),:,:]
        feature_slice = x_feature_shot[:,(self.D3-self.D3//self.s):self.D3,:,:]
        x_feat_long = self.conv_block1(feature)
        feature_concat = torch.cat((feature_slice, x), 1)
        out = x_feat_long + feature_concat
        out = self.compress(out)
        return out


class IDN(nn.Module):
    def __init__(self, color=3,scale=4,rgb_range=3):
        super(IDN, self).__init__()
        nFeat = 64
        nDiff = 16
        nFeat_slice = 4
        nChannel =color
        self.scale = scale
        self.conv1 = nn.Sequential(*[nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(0.05)])
        self.conv2 = nn.Sequential(*[nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(0.05)])

        self.Enhan_unit1 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit2 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit3 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit4 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        # Upsampler
        self.upsample = nn.ConvTranspose2d(nFeat, nChannel, stride=self.scale, kernel_size=17, padding=8)

        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        # x_bicubic = Inter_Bicubic(x, self.scale)
        x_bicubic =F.interpolate(x, scale_factor=self.scale, mode='bicubic',align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.Enhan_unit1(x)
        x = self.Enhan_unit2(x)
        x = self.Enhan_unit3(x)
        x = self.Enhan_unit4(x)

        x_upsample = self.upsample(x, output_size=x_bicubic.size())
        out = x_upsample + x_bicubic
        out = self.add_mean(out)
        return out 