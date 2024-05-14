# Copyright (c) 2023 ✨Challyfilio✨
import torch.nn as nn
import torch
import torch.nn.functional as F

# from conv_block import Conv
import functools
from functools import partial
import os, sys

from typing import List, Tuple, Union
from torch import Tensor

from loguru import logger

from ..builder import ROTATED_NECKS


# from inplace_abn import InPlaceABN, InPlaceABNSync
# from model.sync_batchnorm import SynchronizedBatchNorm2d

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')
# from torch.nn import SyncBatchNorm


class conv_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=(1, 1),
                 group=1,
                 bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=group,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)


@ROTATED_NECKS.register_module()
class BiLSKFPN(nn.Module):
    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            add_extra_convs: Union[bool, str] = False,
    ) -> None:
        super(BiLSKFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs

        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.spfm = SPFM(self.in_channels[3], 2 * self.in_channels[3], 4)  # 最后一层特征图channl

        self.egca1 = EGCA(self.in_channels[0])
        self.egca2 = EGCA(self.in_channels[1])
        self.egca3 = EGCA(self.in_channels[2])
        self.egca4 = EGCA(self.in_channels[3])

        self.d2u2 = D2U(self.in_channels[0], self.in_channels[1])
        self.d2u3 = D2U(self.in_channels[1], self.in_channels[2])
        self.d2u4 = D2U(self.in_channels[2], self.in_channels[3])

        # Decoder-based subpixel convolution
        self.dsc5 = DSCModule(self.in_channels[3], self.in_channels[2])
        self.dsc4 = DSCModule(self.in_channels[2], self.in_channels[1])
        self.dsc3 = DSCModule(self.in_channels[1], self.in_channels[0])
        self.dsc2 = DSCModule(self.in_channels[0], int(self.in_channels[0] / 2))

        self.adjust2 = Adjustment(self.in_channels[0], self.out_channels)
        self.adjust3 = Adjustment(self.in_channels[1], self.out_channels)
        self.adjust4 = Adjustment(self.in_channels[2], self.out_channels)
        self.adjust5 = Adjustment(self.in_channels[3], self.out_channels)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        x2, x3, x4, x5 = inputs

        spfm = self.spfm(x5)  # [2, 4096, 32, 32]

        dsc5 = self.dsc5(x5, spfm)  # [2, 2048, 32, 32]
        dsc4 = self.dsc4(x4, dsc5)  # [2, 1024, 64, 64]
        dsc3 = self.dsc3(x3, dsc4)  # [2, 512, 128, 128]
        dsc2 = self.dsc2(x2, dsc3)  # [2, 256, 256, 256]

        d2u2 = self.d2u2(dsc2)  # [2, 512, 128, 128]
        d2u3 = self.d2u3(d2u2)  # [2, 1024, 64, 64]
        d2u4 = self.d2u4(d2u3)  # [2, 2048, 32, 32]

        # Efficient global context aggregation
        egca5 = self.egca4(x5)  # [2, 2048, 32, 32]
        egca4 = self.egca3(x4)  # [2, 1024, 64, 64]
        egca3 = self.egca2(x3)  # [2, 512, 128, 128]
        egca2 = self.egca1(x2)  # [2, 256, 256, 256]

        adj5 = self.adjust5(egca5 + d2u4)  # [2, 256, 32, 32]
        adj4 = self.adjust4(egca4 + d2u3)  # [2, 256, 64, 64]
        adj3 = self.adjust3(egca3 + d2u2)  # [2, 256, 128, 128]
        adj2 = self.adjust2(egca2 + dsc2)  # [2, 256, 256, 256]

        outs = []
        outs.append(adj2)
        outs.append(adj3)
        outs.append(adj4)
        outs.append(adj5)

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # Faster R-CNN
            if not self.add_extra_convs:
                for i in range(self.num_outs - self.num_ins):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


class D2U(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(D2U, self).__init__()
        self.conv = conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bn_act=True)

    def forward(self, x):
        x = self.conv(x)
        # logger.info(x.shape)
        return x


'''
class RPPModule(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=4,
                       group=1, dilation=4, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=8,
                       group=1, dilation=8, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        br2 = self.conv_dws2(x) # 2023.08.11

        out = torch.cat((br1, br2), dim=1)
        out = self.fusion(out)

        br3 = self.conv_dws3(F.adaptive_avg_pool2d(x, (1, 1)))
        output = br3 + out
        return output
'''


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class SPFM(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super(SPFM, self).__init__()

        assert in_channels % num_splits == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_splits = num_splits
        # self.subspaces = nn.ModuleList(
        #     [RPPModule(int(self.in_channels / self.num_splits)) for i in range(self.num_splits)])
        self.subspaces = nn.ModuleList(
            [LSKblock(int(self.in_channels / self.num_splits)) for i in range(self.num_splits)])
        self.out = conv_block(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x):
        group_size = int(self.in_channels / self.num_splits)
        sub_Feat = torch.chunk(x, self.num_splits, dim=1)
        out = []
        for id, l in enumerate(self.subspaces):
            out.append(self.subspaces[id](sub_Feat[id]))
        out = torch.cat(out, dim=1)
        out = self.out(out)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class EGCA(nn.Module):
    def __init__(self, channels, rate=4):
        super(EGCA, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, int(channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channels / rate), channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, int(channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channels / rate), channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(channels)
        )
        self.branch = nn.Sequential(
            conv_block(channels, channels, kernel_size=3, stride=1, padding=1, group=channels, bn_act=True),
            conv_block(channels, channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x_C = x * x_channel_att
        x_spatial_att = self.spatial_attention(x_C).sigmoid()
        x_S = x * x_spatial_att

        out = torch.add(x_S, x)
        out = channel_shuffle(out, groups=2)

        br = self.branch(x)
        output = br + out
        return output


class DSCModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DSCModule, self).__init__()
        self.conv1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.conv3 = nn.Sequential(
            conv_block(2 * in_channels, 2 * out_channels, kernel_size=3, stride=1, padding=1, bn_act=True),
            nn.PixelShuffle(upscale_factor=1))
        # self.conv3 = conv_block(2 * in_channels, 2 * out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.conv4 = conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x_low, y_high):
        h, w = x_low.size(2), x_low.size(3)

        y_high = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        y_high = self.conv4(y_high)  # 1*1
        y_high = self.conv2(y_high)
        x_low = self.conv1(x_low)

        out = torch.cat([x_low, y_high], 1)
        out = self.conv3(out)
        return out


# 1x1卷积
class Adjustment(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adjustment, self).__init__()
        self.conv = conv_block(in_channels, out_channels, 1, 1, padding=0, bn_act=True)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    input_tensor2 = torch.rand(2, 256, 256, 256)
    input_tensor3 = torch.rand(2, 512, 128, 128)
    input_tensor4 = torch.rand(2, 1024, 64, 64)
    input_tensor5 = torch.rand(2, 2048, 32, 32)
    channels50 = [256, 512, 1024, 2048]  # rn50
    channels34 = [64, 128, 256, 512]  # rn34
    model = BiLSKFPN(in_channels=channels50, out_channels=256, num_outs=5)
    outputs = model((input_tensor2, input_tensor3, input_tensor4, input_tensor5))
    logger.success(len(outputs))
    for j in outputs:
        print(j.shape)
    exit()
