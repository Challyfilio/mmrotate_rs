# Copyright (c) 2023 ✨Challyfilio✨
import torch
import torch.nn as nn
from typing import Sequence, Optional
from ..builder import ROTATED_BACKBONES
from mmdet.models.backbones import SwinTransformer
from collections import OrderedDict
import torch.nn.functional as F

from mmengine.logging import MMLogger
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner.checkpoint import CheckpointLoader


@ROTATED_BACKBONES.register_module()
class SwinTransformerMIM(SwinTransformer):
    # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/swin.py
    """ Swin Transformer
        A PyTorch implement of : `Swin Transformer:
        Hierarchical Vision Transformer using Shifted Windows`  -
            https://arxiv.org/abs/2103.14030

        Inspiration from
        https://github.com/microsoft/Swin-Transformer

        Args:
            pretrain_img_size (int | tuple[int]): The size of input image when
                pretrain. Defaults: 224.
            in_channels (int): The num of input channels.
                Defaults: 3.
            embed_dims (int): The feature dimension. Default: 96.
            patch_size (int | tuple[int]): Patch size. Default: 4.
            window_size (int): Window size. Default: 7.
            mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
                Default: 4.
            depths (tuple[int]): Depths of each Swin Transformer stage.
                Default: (2, 2, 6, 2).
            num_heads (tuple[int]): Parallel attention heads of each Swin
                Transformer stage. Default: (3, 6, 12, 24).
            strides (tuple[int]): The patch merging or patch embedding stride of
                each Swin Transformer stage. (In swin, we set kernel size equal to
                stride.) Default: (4, 2, 2, 2).
            out_indices (tuple[int]): Output from which stages.
                Default: (0, 1, 2, 3).
            qkv_bias (bool, optional): If True, add a learnable bias to query, key,
                value. Default: True
            qk_scale (float | None, optional): Override default qk scale of
                head_dim ** -0.5 if set. Default: None.
            patch_norm (bool): If add a norm layer for patch embed and patch
                merging. Default: True.
            drop_rate (float): Dropout rate. Defaults: 0.
            attn_drop_rate (float): Attention dropout rate. Default: 0.
            drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
            use_abs_pos_embed (bool): If True, add absolute position embedding to
                the patch embedding. Defaults: False.
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='GELU').
            norm_cfg (dict): Config dict for normalization layer at
                output of backone. Defaults: dict(type='LN').
            with_cp (bool, optional): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.
            pretrained (str, optional): model pretrained path. Default: None.
            convert_weights (bool): The flag indicates whether the
                pre-trained model is from the original repo. We may need
                to convert some keys to make it compatible.
                Default: False.
            frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
                Default: -1 (-1 means not freezing any parameters).
            init_cfg (dict, optional): The Config for initialization.
                Defaults to None.
        """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None) -> None:
        super().__init__(
            pretrain_img_size=pretrain_img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_size=patch_size,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            depths=depths,
            num_heads=num_heads,
            strides=strides,
            out_indices=out_indices,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            patch_norm=patch_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_abs_pos_embed=use_abs_pos_embed,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            pretrained=pretrained,
            convert_weights=convert_weights,
            frozen_stages=frozen_stages,
            init_cfg=init_cfg)

    def init_weights(self) -> None:
        """Initialize weights."""
        # super().init_weights()
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                logger.warn("swin_converter")
                _state_dict = swin_converter(_state_dict)

            # logger.warn(_state_dict.items())

            state_dict = OrderedDict()
            # for k, v in _state_dict.items():
            #     if k.startswith('backbone.'):
            #         state_dict[k[9:]] = v

            # strip prefix of state_dict
            # if list(state_dict.keys())[0].startswith('module.'):
            #     state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        trunc_normal_(self.mask_token, mean=0, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def swin_converter(ckpt):
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v
        # new_ckpt[new_k] = new_v

    return new_ckpt
