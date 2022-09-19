import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(modules):
    pass


class MeanShift(nn.Conv2d):
    def __init__(self, mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


class UpSampler(nn.Module):
    def __init__(self, upscale_factor, num_fea=64):
        super(UpSampler, self).__init__()
        if (upscale_factor & (upscale_factor-1)) == 0: # upscale_factor = 2^n
            m = []
            for i in range(int(math.log(upscale_factor, 2))):
                m.append(nn.Conv2d(num_fea, num_fea * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            self.upsample = nn.Sequential(*m)

        elif upscale_factor == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_fea, num_fea * 9, 3, 1, 1),
                nn.PixelShuffle(3)
            )
        else:
            raise NotImplementedError('Error upscale_factor in Upsampler')

    def forward(self, x):
        return self.upsample(x)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out