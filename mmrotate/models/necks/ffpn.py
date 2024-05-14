# Copyright (c) 2023 ✨Challyfilio✨
# FFPN
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import ROTATED_NECKS
from mmdet.models.necks.fpn import FPN
from loguru import logger


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class DSCModule(nn.Module):
    def __init__(self, in_channels, out_channels, red=1):
        super(DSCModule, self).__init__()

        self.conv1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        # self.conv2 = conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.conv3 = nn.Sequential(
            # conv_block(2 * in_channels, 2 * out_channels, kernel_size=3, stride=1, padding=1, bn_act=True),
            nn.PixelShuffle(upscale_factor=2),
            conv_block(int(in_channels / 2), out_channels, kernel_size=3, stride=2, padding=1, bn_act=True)
        )

    def forward(self, x_gui, y_high):
        # logger.info('low: ' + str(x_gui.shape))
        # logger.info('high: ' + str(y_high.shape))
        h, w = x_gui.size(2), x_gui.size(3)

        y_high = F.interpolate(y_high, size=(h, w), mode='nearest')
        # y_high = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        # y_high = self.conv2(y_high)

        out = torch.cat([y_high, x_gui], 1)
        # logger.info(out.shape)

        out = self.conv3(out)
        # logger.info(out.shape)
        return out


@ROTATED_NECKS.register_module()
class FFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        self.dsc_models = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            dsc_model = DSCModule(out_channels, out_channels)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
            self.dsc_models.append(dsc_model)

    def forward(self, inputs):
        # for inp in inputs:
        #     logger.info(inp.shape)
        # exit()
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # print(len(self.downsample_convs))  # 3
        # print(len(self.pafpn_convs))  # 3
        # print(len(self.dsc_models))  # 3

        # build laterals [256,128,64,32]
        laterals = [
            # 统一输出通道数 1*1卷积
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # for l in laterals:
        #     logger.info(l.shape)
        # logger.info(len(laterals)) # 4

        # build top-down path
        used_backbone_levels = len(laterals)  # 4
        for i in range(used_backbone_levels - 1, 0, -1):  # i = 3,2,1
            # 上采样
            # prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] = laterals[i - 1] + F.interpolate(
            #     laterals[i], size=prev_shape, mode='nearest')
            # logger.info(i)
            laterals[i - 1] = self.dsc_models[i - 1](laterals[i - 1], laterals[i])

        # build outputs
        inter_outs = laterals  # DSC输出
        # part 1: from original levels
        # inter_outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):  # i = 0,1,2
            # 下采样
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)  # i = 1,2,3
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


if __name__ == "__main__":
    input_tensor2 = torch.rand(2, 768, 256, 256)
    input_tensor3 = torch.rand(2, 768, 128, 128)
    input_tensor4 = torch.rand(2, 768, 64, 64)
    input_tensor5 = torch.rand(2, 768, 32, 32)
    channels50 = [256, 512, 1024, 2048]  # rn50
    channels34 = [64, 128, 256, 512]  # rn34
    channels768 = [768, 768, 768, 768]  # rn34
    model = FFPN(in_channels=channels50, out_channels=256, num_outs=5)
    outputs = model((input_tensor2, input_tensor3, input_tensor4, input_tensor5))
    logger.success(len(outputs))
    for j in outputs:
        print(j.shape)
    exit()
