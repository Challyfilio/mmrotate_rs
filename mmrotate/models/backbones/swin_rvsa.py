# Copyright (c) 2023 ✨Challyfilio✨
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple

from einops import rearrange, repeat
# from mmdet.registry import MODELS
from mmrotate.models.layers import PatchEmbed, PatchMerging

from loguru import logger


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, H, W, mask=None):
        logger.debug(x.shape)
        # logger.debug(mask.shape)
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        logger.debug(x.shape)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


# ---- RVSA
class RotatedVariedSizeWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, out_dim=None, window_size_rvsa=1, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0, attn_head_dim=None, relative_pos_embedding=True, learnable=True,
                 restart_regression=True,
                 attn_window_size=None, shift_size=0, img_size=(1, 1), num_deform=None):
        super().__init__()

        # window_size_rvsa = window_size_rvsa[0]
        window_size_rvsa = window_size_rvsa

        self.img_size = to_2tuple(img_size)
        self.num_heads = num_heads
        self.dim = dim
        out_dim = out_dim or dim
        self.out_dim = out_dim
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size_rvsa
        attn_window_size = attn_window_size or window_size_rvsa
        self.attn_ws = attn_window_size or self.ws

        q_size = window_size_rvsa
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            # if num_deform is None, we set num_deform to num_heads as default

            if num_deform is None:
                num_deform = 1
            self.num_deform = num_deform

            self.sampling_offsets = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size_rvsa, stride=window_size_rvsa),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            self.sampling_scales = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size_rvsa, stride=window_size_rvsa),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            # add angle
            self.sampling_angles = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size_rvsa, stride=window_size_rvsa),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 1, kernel_size=1, stride=1)
            )

        self.shift_size = shift_size % self.ws
        # self.left_size = self.img_size
        #        if min(self.img_size) <= self.ws:
        #            self.shift_size = 0

        # if self.shift_size > 0:
        #     self.padding_bottom = (self.ws - self.shift_size + self.padding_bottom) % self.ws
        #     self.padding_right = (self.ws - self.shift_size + self.padding_right) % self.ws

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, out_dim * 3, bias=qkv_bias)

        # self.qkv = nn.Conv2d(dim, out_dim * 3, 1, bias=qkv_bias)
        # self.kv = nn.Conv2d(dim, dim*2, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        # self.proj = nn.Conv2d(out_dim, out_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size_rvsa + attn_window_size - 1) * (window_size_rvsa + attn_window_size - 1),
                            num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.attn_ws)
            coords_w = torch.arange(self.attn_ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.attn_ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.attn_ws - 1
            relative_coords[:, :, 0] *= 2 * self.attn_ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x, H, W):

        B, N, C = x.shape
        assert N == H * W
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        b, _, h, w = x.shape
        shortcut = x
        # assert h == self.img_size[0]
        # assert w == self.img_size[1]
        # if self.shift_size > 0:
        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left

        # padding on left-right-up-down
        expand_h, expand_w = h + padding_top + padding_down, w + padding_left + padding_right

        # window num in padding features
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws

        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2,
                                                                                                       1).unsqueeze(
            0)  # 1, 2, H, W

        # position of the window relative to the image center
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)  # 1,2, nh, nw
        image_reference = image_reference.reshape(1, 2, window_num_h, self.ws, window_num_w,
                                                  self.ws)  # 1, 2, nh, ws, nw, ws
        assert window_num_h == window_reference.shape[-2]
        assert window_num_w == window_reference.shape[-1]

        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)  # 1,2, nh,1, nw,1

        # coords of pixels in each window

        base_coords_h = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_h - 1)  # ws
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_w - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())
        # base_coords = torch.stack(torch.meshgrid(base_coords_w, base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, 1, self.attn_ws, 1, self.attn_ws)

        # extend to each window
        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)  # ws -> 1,ws -> nh,ws
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.attn_ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)  # nw,ws
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.attn_ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)  # nh*ws
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)  # nw*ws

        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
                                                                                                               1).reshape(
            1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws)  # 1, 2, nh, ws, nw, ws
        # base_coords = window_reference+window_coords
        base_coords = image_reference

        # padding feature
        x = torch.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))

        if self.restart_regression:
            # compute for each head in each batch
            coords = base_coords.repeat(b * self.num_heads, 1, 1, 1, 1, 1)  # B*nH, 2, nh, ws, nw, ws
        if self.learnable:
            # offset factors
            sampling_offsets = self.sampling_offsets(x)

            num_predict_total = b * self.num_heads * self.num_deform

            sampling_offsets = sampling_offsets.reshape(num_predict_total, 2, window_num_h, window_num_w)
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (h // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (w // self.ws)

            # scale fators
            sampling_scales = self.sampling_scales(x)  # B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)

            # rotate factor
            sampling_angle = self.sampling_angles(x)
            sampling_angle = sampling_angle.reshape(num_predict_total, 1, window_num_h, window_num_w)

            # first scale

            window_coords = window_coords * (sampling_scales[:, :, :, None, :, None] + 1)

            # then rotate around window center

            window_coords_r = window_coords.clone()

            # 0:x,column, 1:y,row

            window_coords_r[:, 0, :, :, :, :] = -window_coords[:, 1, :, :, :, :] * torch.sin(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * torch.cos(
                sampling_angle[:, 0, :, None, :, None])
            window_coords_r[:, 1, :, :, :, :] = window_coords[:, 1, :, :, :, :] * torch.cos(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * torch.sin(
                sampling_angle[:, 0, :, None, :, None])

            # system transformation: window center -> image center

            coords = window_reference + window_coords_r + sampling_offsets[:, :, :, None, :, None]

        # final offset
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, self.attn_ws * window_num_h,
                                                                 self.attn_ws * window_num_w, 2)

        qkv = self.qkv(shortcut.permute(0, 2, 3, 1).reshape(b, -1, self.dim)).permute(0, 2, 1).reshape(b, -1, h,
                                                                                                       w).reshape(b, 3,
                                                                                                                  self.num_heads,
                                                                                                                  self.out_dim // self.num_heads,
                                                                                                                  h,
                                                                                                                  w).transpose(
            1, 0).reshape(3 * b * self.num_heads, self.out_dim // self.num_heads, h, w)

        qkv = torch.nn.functional.pad(qkv, (padding_left, padding_right, padding_top, padding_down)).reshape(3,
                                                                                                             b * self.num_heads,
                                                                                                             self.out_dim // self.num_heads,
                                                                                                             h + padding_td,
                                                                                                             w + padding_lr)

        q, k, v = qkv[0], qkv[1], qkv[2]  # b*self.num_heads, self.out_dim // self.num_heads, H，W

        k_selected = F.grid_sample(
            k.reshape(num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td,
                      w + padding_lr),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr)
        v_selected = F.grid_sample(
            v.reshape(num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td,
                      w + padding_lr),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr)

        q = q.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.ws, window_num_w,
                      self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w, self.num_heads,
                                                                    self.ws * self.ws, self.out_dim // self.num_heads)
        k = k_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws,
                               window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(
            b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
            self.out_dim // self.num_heads)
        v = v_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws,
                               window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(
            b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
            self.out_dim // self.num_heads)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        dots = calc_rel_pos_spatial(dots, q, (self.ws, self.ws), (self.attn_ws, self.attn_ws), self.rel_pos_h,
                                    self.rel_pos_w)

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.attn_ws * self.attn_ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                        hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        # if self.shift_size > 0:
        # out = torch.masked_select(out, self.select_mask).reshape(b, -1, h, w)
        out = out[:, :, padding_top:h + padding_top, padding_left:w + padding_left]

        out = out.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    def _clip_grad(self, grad_norm):
        # print('clip grads of the model for selection')
        nn.utils.clip_grad_norm_(self.sampling_offsets.parameters(), grad_norm)
        nn.utils.clip_grad_norm_(self.sampling_scales.parameters(), grad_norm)

    def _reset_parameters(self):
        if self.learnable:
            nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
            nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
            nn.init.constant_(self.sampling_scales[-1].weight, 0.)
            nn.init.constant_(self.sampling_scales[-1].bias, 0.)

    def flops(self, ):
        N = self.ws * self.ws
        M = self.attn_ws * self.attn_ws
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * M
        #  x = (attn @ v)
        flops += self.num_heads * N * M * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        h, w = self.img_size[0] + self.shift_size + self.padding_bottom, self.img_size[
            1] + self.shift_size + self.padding_right
        flops *= (h / self.ws * w / self.ws)

        # for sampling
        flops_sampling = 0
        if self.learnable:
            # pooling
            flops_sampling += h * w * self.dim
            # regressing the shift and scale
            flops_sampling += 2 * (h / self.ws + w / self.ws) * self.num_heads * 2 * self.dim
            # calculating the coords
            flops_sampling += h / self.ws * self.attn_ws * w / self.ws * self.attn_ws * 2
        # grid sampling attended features
        flops_sampling += h / self.ws * self.attn_ws * w / self.ws * self.attn_ws * self.dim

        flops += flops_sampling

        return flops


# ----

class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        # self.w_msa = RotatedVariedSizeWindowAttention(
        #     dim=embed_dims,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop_rate,
        #     proj_drop=proj_drop_rate,
        #     window_size_rvsa=window_size)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape  # ✨Challyfilio✨
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        # attn_windows = self.w_msa(query_windows, mask=attn_mask)
        # attn_windows = self.w_msa(query_windows, H, W, mask=attn_mask)
        attn_windows = self.w_msa(query_windows, H, W)  # rvsa

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


# @MODELS.register_module()
class SwinTransformerRVSA(BaseModule):
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
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformerRVSA, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformerRVSA, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i - 1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
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
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

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

    def forward(self, x):
        # B, C, H, W = x.shape
        x, hw_shape = self.patch_embed(x)
        logger.info(hw_shape)
        # Hp, Wp = hw_shape

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs


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

    return new_ckpt


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
            torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )

    # dist_h [1-ws,ws-1]->[0,2ws-2]

    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
            torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # get pos encode, qwh, kwh, C'

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)  # B, H, qwh, qww, C
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)  # B, H, qwh, qww, C'; qwh, kWh, C' -> B,H,qwh,qww,kwh
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)  # B,H,qwh,qww,kww

    # attn: B,H,qwh,qww,kwh,kww

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


if __name__ == '__main__':
    model = SwinTransformerRVSA(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True)
    input = torch.rand(1, 3, 1024, 1024)  # B,C,H,W
    output = logger.success(len(model(input)))
