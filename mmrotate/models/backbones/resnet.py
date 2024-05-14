# Copyright (c) 2023 ✨Challyfilio✨
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

import math
from mmcv.ops import DeformConv2d
# from mmcv.ops.deform_conv import DeformConv2d, DeformConv2dFunction
from mmcv.utils import print_log
from torch.nn.modules.utils import _pair, _single

from loguru import logger
from ..builder import ROTATED_BACKBONES
# from mmdet.registry import MODELS
from ..layers import ResLayer


# deform_conv2d = DeformConv2dFunction.apply


# # # # # # # # # # #
# 坐标相加
def pos_add(a, b):
    xa, ya = a
    xb, yb = b
    return (xa + xb, ya + yb)


def pos_sub(a, b):
    xa, ya = a
    xb, yb = b
    return (xa - xb, ya - yb)


def pos_opposite(a):
    xa, ya = a
    return (-xa, -ya)


def zhongdian(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2,


def stretch(x1, y1, x2, y2, x3, y3, x4, y4):
    # 中点坐标
    gx, gy = zhongdian(x1, y1, x3, y3)
    # print(gx, gy)

    # 计算两条对角线的长度
    d_AC = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)  # AC长度
    d_BD = math.sqrt((x4 - x2) ** 2 + (y4 - y2) ** 2)  # BD长度

    # 判断哪条是较短的对角线，并求出另外两个顶点坐标
    if d_AC < d_BD:
        # print('AC<BD')
        # AC是较短的对角线
        stat = 'BD'
        x6 = d_BD / d_AC * (x3 - gx) + gx  # F点横坐标
        y6 = d_BD / d_AC * (y3 - gy) + gy  # F点纵坐标
    else:
        # print('AC>BD')
        # BD是较短的对角线
        stat = 'AC'
        x6 = d_AC / d_BD * (x4 - gx) + gx  # F点横坐标
        y6 = d_AC / d_BD * (y4 - gy) + gy  # F点纵坐标
    x5 = 2 * gx - x6  # E点横坐标
    y5 = 2 * gy - y6  # E点纵坐标
    return x5, y5, x6, y6, stat


def cal_offset(ofs1, ofs2):
    pn = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    out_offset = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

    x1, y1 = pos_add(pn[0], ofs1)  # A点
    # x2, y2 = pos_add(pn[2], pos_opposite(ofs))  # B点
    # x3, y3 = pos_add(pn[8], pos_opposite(ofs))  # C点
    x2, y2 = pos_add(pn[2], ofs2)  # B点
    x3, y3 = pos_add(pn[8], ofs2)  # C点
    x4, y4 = pos_add(pn[6], ofs1)  # D点
    # print((x1, y1), (x2, y2), (x3, y3), (x4, y4))

    x5, y5, x6, y6, stat = stretch(x1, y1, x2, y2, x3, y3, x4, y4)  # 拉伸后E、F点, stat为长的对角线
    # 输出结果
    # print("另外两个顶点坐标为：")
    # print("E ({:.2f}, {:.2f})".format(x5, y5))
    # print("F ({:.2f}, {:.2f})".format(x6, y6))

    if stat == 'AC':
        out_offset[0] = ofs1
        out_offset[2] = pos_sub((x5, y5), pn[2])
        out_offset[6] = pos_sub((x6, y6), pn[6])
        # out_offset[8] = pos_opposite(ofs)
        out_offset[8] = ofs2
        # print('# ' * 10)
        # print((x1, y1), (x5, y5), (x3, y3), (x6, y6))
        out_offset[1] = pos_sub(zhongdian(x1, y1, x5, y5), pn[1])
        out_offset[3] = pos_sub(zhongdian(x1, y1, x6, y6), pn[3])
        out_offset[5] = pos_sub(zhongdian(x5, y5, x3, y3), pn[5])
        out_offset[7] = pos_sub(zhongdian(x3, y3, x6, y6), pn[7])
        # 中点偏移
        out_offset[4] = pos_sub(zhongdian(x5, y5, x6, y6), pn[4])
    else:
        out_offset[6] = ofs1
        # out_offset[2] = pos_opposite(ofs)
        out_offset[2] = ofs2
        out_offset[0] = pos_sub((x5, y5), pn[0])
        out_offset[8] = pos_sub((x6, y6), pn[8])
        # print('# ' * 10)
        # print((x5, y5), (x2, y2), (x6, y6), (x4, y4))
        out_offset[1] = pos_sub(zhongdian(x5, y5, x2, y2), pn[1])
        out_offset[3] = pos_sub(zhongdian(x5, y5, x4, y4), pn[3])
        out_offset[5] = pos_sub(zhongdian(x2, y2, x6, y6), pn[5])
        out_offset[7] = pos_sub(zhongdian(x6, y6, x4, y4), pn[7])
        # 中点偏移
        out_offset[4] = pos_sub(zhongdian(x5, y5, x6, y6), pn[4])

    # print('- ' * 10)
    # offx_list = []
    # offy_list = []
    # for i, j in zip(pn, out_offset):
    #     offx_list.append(pos_add(i, j)[0])
    #     offy_list.append(pos_add(i, j)[1])
    #     print(pos_add(i, j))

    outs = []
    # for i, j in zip(pn, out_offset):
    #     temp = pos_add(i, j)
    #     outs.append(temp[0])
    #     outs.append(temp[1])
    for o in out_offset:
        outs.append(o[0])
        outs.append(o[1])
    return outs


# # # # # # # # # # # # #
'''
class DeformConv2dPack1(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.conv_offset = nn.Conv2d(
        #     self.in_channels,
        #     # self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
        #     self.deform_groups * 2,
        #     kernel_size=self.kernel_size,
        #     stride=_pair(self.stride),
        #     padding=_pair(self.padding),
        #     dilation=_pair(self.dilation),
        #     bias=True)
        # self.init_offset()
        self.offset1 = nn.Parameter(torch.zeros(2, ))

    # def init_offset(self):
    #     self.conv_offset.weight.data.zero_()
    #     self.conv_offset.bias.data.zero_()

    def forward(self, x):  # type: ignore
        batchsize, H, W = x.shape[0], x.shape[2], x.shape[3]
        logger.error(batchsize)
        logger.error(H)
        logger.error(W)

        # self.offset1 = self.offset1.cpu()
        offset1_list = self.offset1.tolist()
        offset1_tuple = tuple(offset1_list)
        # logger.debug(offset1_tuple)
        out_offset = cal_offset(offset1_tuple)
        # logger.debug(out_offset)
        offsets = torch.tensor(out_offset)
        # logger.info(offsets)
        offsets = offsets.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # logger.info(offsets)
        offsets = offsets.expand(batchsize, -1, H, W).requires_grad_(True).cuda()
        # offsets = offsets.repeat(batchsize, 1, H, W).requires_grad_(True).cuda()
        # offsets = offsets.view(batchsize, 18, H, W)
        logger.debug(offsets.shape)
        offsets = offsets.contiguous()
        # logger.info(offsets)
        # logger.error('- - - - - - - - - -')

        # offset = self.conv_offset(x)
        # logger.debug(self.offset1.shape)  # torch.Size([2, 2, 256, 256])
        # logger.debug(offset)
        return deform_conv2d(x, offsets, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        # if version is None or version < 2:
        #     # the key is different in early versions
        #     # In version < 2, DeformConvPack loads previous benchmark models.
        #     if (prefix + 'conv_offset.weight' not in state_dict
        #             and prefix[:-1] + '_offset.weight' in state_dict):
        #         state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
        #             prefix[:-1] + '_offset.weight')
        #     if (prefix + 'conv_offset.bias' not in state_dict
        #             and prefix[:-1] + '_offset.bias' in state_dict):
        #         state_dict[prefix +
        #                    'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
        #                                                         '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
'''


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.tag = 'norm'
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
            # logger.success(self.conv2_stride)
            # self.conv2 = DeformConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=dilation, dilation=dilation,
            #                           deform_groups=1)
            #####################
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            # self.conv2 = build_conv_layer(
            #     dcn,
            #     planes,
            #     planes,
            #     kernel_size=3,
            #     stride=self.conv2_stride,
            #     padding=dilation,
            #     dilation=dilation,
            #     bias=False)
            ######################
            self.tag = 'dcn'
            self.conv2 = DeformConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=dilation, dilation=dilation,
                                      deform_groups=1)

        self.add_module(self.norm2_name, norm2)
        if self.tag == 'norm':
            self.conv3 = build_conv_layer(
                conv_cfg,
                planes,
                planes * self.expansion,
                kernel_size=1,
                bias=False)
        else:
            self.conv3 = build_conv_layer(
                conv_cfg,
                planes,
                planes * self.expansion,
                stride=self.conv2_stride,
                kernel_size=1,
                bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

        self.offset = nn.Parameter(torch.zeros(4, )).to('cuda')
        offset1_list = self.offset.tolist()
        # offset1_tuple = tuple(offset1_list)
        offset1_tuple = (offset1_list[0], offset1_list[1])
        offset2_tuple = (offset1_list[2], offset1_list[3])
        out_offset = cal_offset(offset1_tuple, offset2_tuple)
        offsets = torch.tensor(out_offset).to('cuda')
        self.offsets = offsets.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            self.offsets = self.offsets.expand(x.shape[0], -1, x.shape[2], x.shape[3])
            # offsets = offsets.repeat(batchsize, 1, H, W).requires_grad_(True).cuda()
            # offsets = offsets.view(batchsize, 18, H, W)
            # logger.debug(self.offsets.shape)
            self.offsets = self.offsets.contiguous()
            # logger.info(offsets)
            # logger.error('- - - - - - - - - -')

            if self.tag == 'norm':
                # logger.error('norm')
                out = self.conv2(out)
            else:
                # logger.error('dcn')
                out = self.conv2(out, self.offsets)
            # logger.error(out.shape)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)
                # logger.error('1 ' + str(identity.shape))

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@ROTATED_BACKBONES.register_module()
# @MODELS.register_module()
class ResNetDC(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ResNetDC, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (
                len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                # print(x.shape)
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNetDC, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


# @MODELS.register_module()
@ROTATED_BACKBONES.register_module()
class ResNetDCV1d(ResNetDC):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetDCV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
