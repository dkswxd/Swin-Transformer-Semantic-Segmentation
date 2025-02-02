# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.utils import get_root_logger
from ..builder import BACKBONES

from mmcv.parallel import is_module_wrapper
from mmcv.runner import get_dist_info


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    pretrain = torch.load(filename, map_location=map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(pretrain, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in pretrain:
        state_dict = pretrain['state_dict']
    elif 'model' in pretrain:
        state_dict = pretrain['model']
    else:
        state_dict = pretrain
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # reshape absolute position embedding
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

    # interpolate position bias table if needed
    modified_key = ['qkv.weight', 'qkv.bias', 'proj.weight', 'proj.bias']
    for layer_i, layer in enumerate(model.layers):
        for blocks_i, block in enumerate(layer.blocks):
            table_key = f'layers.{layer_i}.blocks.{blocks_i}.attn.relative_position_bias_table'
            table_pretrained = state_dict[table_key]
            L1, nH1 = table_pretrained.size()
            S1 = int(L1 ** 0.5)
            for attn_i, attn in enumerate(block.attn):
                for m_key in modified_key:
                    state_dict[f'layers.{layer_i}.blocks.{blocks_i}.attn.{attn_i}.{m_key}'] = \
                        state_dict[f'layers.{layer_i}.blocks.{blocks_i}.attn.{m_key}']
                table_key_current = f'layers.{layer_i}.blocks.{blocks_i}.attn.{attn_i}.relative_position_bias_table'
                table_current = state_dict[table_key_current]
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f"Error in loading {table_key}, pass")
                else:
                    if L1 != L2:
                        S2_S, S2_H, S2_W = attn.window_size
                        table_pretrained_resized = F.interpolate(
                            state_dict[table_key].permute(1, 0).view(1, nH1, S1, S1),
                            size=(S2_H, S2_W), mode='bicubic')
                        table_pretrained_resized = table_pretrained_resized.repeat(S2_S, 1, 1, 1, 1).permute(1, 2, 0, 3,
                                                                                                             4).contiguous()
                        state_dict[table_key_current] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
                    else:
                        state_dict[table_key_current] = state_dict[table_key]

            for m_key in modified_key:
                state_dict.pop(f'layers.{layer_i}.blocks.{blocks_i}.attn.{m_key}')
            state_dict.pop(table_key)
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, S, H, W, C)
        window_size (tuple(int)): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuplr(int)): Window size
        S (int): Spectral of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, S, H, W, C)
    """
    B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, S // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x

def dilate_window_partition(x, window_size, dliate_size):
    """
    Args:
        x: (B, S, H, W, C)
        window_size (tuple(int)): window size
        dliate_size (tuple(int)): dliate size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, S, H, W, C = x.shape
    x = x.view(B,
               S // dliate_size[0] // window_size[0], dliate_size[0], window_size[0],
               H // dliate_size[1] // window_size[1], dliate_size[1], window_size[1],
               W // dliate_size[2] // window_size[2], dliate_size[2], window_size[2],
               C)
    # windows = torch.einsum('abcdefghijk->acfibehdgjk', x).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    windows = x.permute(0, 2, 5, 8, 1, 4, 7, 3, 6, 9, 10).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    # x = x.view(B, S // dliate_size[0], dliate_size[0], H // dliate_size[1], dliate_size[1], W // dliate_size[2], dliate_size[2], C)
    # x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    # x = x.view(B * dliate_size[0] * dliate_size[1] * dliate_size[2], S // dliate_size[0], H // dliate_size[1], W // dliate_size[2], C)
    # x = x.view(B * dliate_size[0] * dliate_size[1] * dliate_size[2],
    #            S // dliate_size[0] // window_size[0], window_size[0],
    #            H // dliate_size[1] // window_size[1], window_size[1],
    #            W // dliate_size[2] // window_size[2], window_size[2],
    #            C)
    # windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def dilate_window_reverse(windows, window_size, dliate_size, S, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuplr(int)): Window size
        dliate_size (tuple(int)): dliate size
        S (int): Spectral of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, S, H, W, C)
    """
    C = windows.shape[-1]
    B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, dliate_size[0], dliate_size[1], dliate_size[2],
                     S // dliate_size[0] // window_size[0],
                     H // dliate_size[1] // window_size[1],
                     W // dliate_size[2] // window_size[2],
                     window_size[0], window_size[1], window_size[2], C)
    # x = torch.einsum('abcdefghijk->aebhfcigdjk', x).contiguous().view(B, S, H, W, C)
    x = x.permute(0, 4, 1, 7, 5, 2, 8, 6, 3, 9, 10).contiguous().view(B, S, H, W, C)
    # x = windows.view(B * dliate_size[0] * dliate_size[1] * dliate_size[2],
    #                  S // dliate_size[0] // window_size[0],
    #                  H // dliate_size[1] // window_size[1],
    #                  W // dliate_size[2] // window_size[2],
    #                  window_size[0], window_size[1], window_size[2], C)
    # x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    # x = x.view(B, dliate_size[0], dliate_size[1], dliate_size[2], S // dliate_size[0], H // dliate_size[1], W // dliate_size[2], C)
    # x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, S, H, W, C)
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, shift_size, dilate, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Ws, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.shift_size=shift_size
        self.dilate=dilate
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Ws-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))  # 3, Ws, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Ws*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Ws*Wh*Ww, Ws*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ws*Wh*Ww, Ws*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Ws*Wh*Ww, Ws*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B, L, C = x.shape
        assert L == S * H * W, "input feature has wrong size"

        x = x.view(B, S, H, W, C)
        true_size = [_window_size * _dilate for (_window_size, _dilate) in zip(self.window_size, self.dilate)]
        true_shift = [_shift_size * _dilate for (_shift_size, _dilate) in zip(self.shift_size, self.dilate)]

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_u = 0
        pad_r = (true_size[2] - W % true_size[2]) % true_size[2]
        pad_b = (true_size[1] - H % true_size[1]) % true_size[1]
        pad_d = (true_size[0] - S % true_size[0]) % true_size[0]
        x = F.pad(x, [0, 0, pad_l, pad_r, pad_t, pad_b, pad_u, pad_d])
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            shifted_x = torch.roll(x, shifts=[-_true_shift for _true_shift in true_shift], dims=(1, 2, 3))
            attn_mask = None
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = dilate_window_partition(shifted_x, self.window_size, self.dilate)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C




        #############################################
        B_, N, C = x_windows.shape
        qkv = self.qkv(x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Ws*Wh*Ww,Ws*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Ws*Wh*Ww, Ws*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        attn_windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        attn_windows = self.proj(attn_windows)
        attn_windows = self.proj_drop(attn_windows)
        #############################################

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = dilate_window_reverse(attn_windows, self.window_size, self.dilate, Sp, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            x = torch.roll(shifted_x, shifts=[_true_shift for _true_shift in true_shift], dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_d > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple(int)): Window size.
        shift_size (tuple(int)): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(4, 4, 4), shift_size=(2, 2, 2), dilate=(1, 2, 2),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.dilate = dilate

        self.norm1 = norm_layer(dim)
        self.attn = nn.ModuleList()
        for ws, ss, d in zip(window_size, shift_size, dilate):
            self.attn.append(WindowAttention(
                dim, window_size=ws, shift_size=ss, dilate=d, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.S = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        S, H, W = self.S, self.H, self.W
        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # W-MSA/SW-MSA
        x_msa = []
        for attn_layer in self.attn:
            x_msa.append(attn_layer(x, S, H, W))  # nW*B, window_size*window_size, C

        # FFN
        x = shortcut
        for x_ in x_msa:
            x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, down_sample_size=(1, 2, 2)):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(down_sample_size[0] * down_sample_size[1] * down_sample_size[2] * dim, 2 * dim, bias=False)
        self.norm = norm_layer(down_sample_size[0] * down_sample_size[1] * down_sample_size[2] * dim)
        self.down_sample_size = down_sample_size

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, S*H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == S * H * W, "input feature has wrong size"

        x = x.view(B, S, H, W, C)
        if self.down_sample_size == (1, 2, 2):
            # padding
            pad_input = (H % 2 == 1) or (W % 2 == 1)
            if pad_input:
                x = F.pad(x, [0, 0, 0, W % 2, 0, H % 2,])

            x0 = x[:, :, 0::2, 0::2, :]  # B S H/2 W/2 C
            x1 = x[:, :, 1::2, 0::2, :]  # B S H/2 W/2 C
            x2 = x[:, :, 0::2, 1::2, :]  # B S H/2 W/2 C
            x3 = x[:, :, 1::2, 1::2, :]  # B S H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B S H/2 W/2 8*C
            x = x.view(B, -1, 4 * C)  # B S*H/2*W/2 4*C
        elif self.down_sample_size == (2, 2, 2):
            # padding
            pad_input = (H % 2 == 1) or (W % 2 == 1) or (S % 2 == 1)
            if pad_input:
                x = F.pad(x, [0, 0, 0, W % 2, 0, H % 2, 0, S % 2])

            x0 = x[:, 0::2, 0::2, 0::2, :]  # B S/2 H/2 W/2 C
            x1 = x[:, 0::2, 1::2, 0::2, :]  # B S/2 H/2 W/2 C
            x2 = x[:, 0::2, 0::2, 1::2, :]  # B S/2 H/2 W/2 C
            x3 = x[:, 0::2, 1::2, 1::2, :]  # B S/2 H/2 W/2 C
            x0_ = x[:, 1::2, 0::2, 0::2, :]  # B S/2 H/2 W/2 C
            x1_ = x[:, 1::2, 1::2, 0::2, :]  # B S/2 H/2 W/2 C
            x2_ = x[:, 1::2, 0::2, 1::2, :]  # B S/2 H/2 W/2 C
            x3_ = x[:, 1::2, 1::2, 1::2, :]  # B S/2 H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3, x0_, x1_, x2_, x3_], -1)  # B S/2 H/2 W/2 8*C
            x = x.view(B, -1, 8 * C)  # B S/2*H/2*W/2 4*C


        x = self.norm(x)
        x = self.reduction(x)

        return x

# class PatchMerging(nn.Module):
#     """ Patch Merging Layer
#
#     Args:
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(8 * dim)
#
#     def forward(self, x, S, H, W):
#         """ Forward function.
#
#         Args:
#             x: Input feature, tensor size (B, S*H*W, C).
#             H, W: Spatial resolution of the input feature.
#         """
#         B, L, C = x.shape
#         assert L == S * H * W, "input feature has wrong size"
#
#         x = x.view(B, S, H, W, C)
#
#         # padding
#         pad_input = (H % 2 == 1) or (W % 2 == 1) or (S % 2 == 1)
#         if pad_input:
#             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, S % 2))
#
#         x0 = x[:, 0::2, 0::2, 0::2, :]  # B S/2 H/2 W/2 C
#         x1 = x[:, 0::2, 1::2, 0::2, :]  # B S/2 H/2 W/2 C
#         x2 = x[:, 0::2, 0::2, 1::2, :]  # B S/2 H/2 W/2 C
#         x3 = x[:, 0::2, 1::2, 1::2, :]  # B S/2 H/2 W/2 C
#         x0_ = x[:, 1::2, 0::2, 0::2, :]  # B S/2 H/2 W/2 C
#         x1_ = x[:, 1::2, 1::2, 0::2, :]  # B S/2 H/2 W/2 C
#         x2_ = x[:, 1::2, 0::2, 1::2, :]  # B S/2 H/2 W/2 C
#         x3_ = x[:, 1::2, 1::2, 1::2, :]  # B S/2 H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3, x0_, x1_, x2_, x3_], -1)  # B S/2 H/2 W/2 8*C
#         x = x.view(B, -1, 8 * C)  # B S/2*H/2*W/2 4*C
#
#         x = self.norm(x)
#         x = self.reduction(x)
#
#         return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple(int)): Local window size. Default: (4, 4, 4).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 dilate,
                 shift,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 down_sample_size=(1, 2, 2),
                 use_checkpoint=False):
        super().__init__()
        assert len(window_size) == len(dilate)
        assert len(window_size[0]) == len(dilate[0])
        self.window_size = window_size
        self.shift = shift
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.down_sample_size = down_sample_size
        self.dilate = dilate
        self.group_len = len(dilate)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift,
                dilate=dilate,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, down_sample_size=down_sample_size)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, S*H*W, C).
            H, W: Spatial resolution of the input feature.
        """


        for blk in self.blocks:
            blk.S, blk.H, blk.W = S, H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            if self.down_sample_size == (1, 2, 2):
                Ws, Wh, Ww = S, (H + 1) // 2, (W + 1) // 2
            elif self.down_sample_size == (2, 2, 2):
                Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (tuple[int]): Patch token size. Default: (4, 4, 4).
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        # if not isinstance(patch_size, tuple):
        #     patch_size = (patch_size, patch_size, patch_size)
        self._patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self._patch_size[2] != 0:
            x = F.pad(x, [0, self._patch_size[2] - W % self._patch_size[2]])
        if H % self._patch_size[1] != 0:
            x = F.pad(x, [0, 0, 0, self._patch_size[1] - H % self._patch_size[1]])
        if S % self._patch_size[0] != 0:
            x = F.pad(x, [0, 0, 0, 0, 0, self._patch_size[0] - S % self._patch_size[0]])

        x = self.proj(x)  # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Ws, Wh, Ww)

        return x


@BACKBONES.register_module()
class SwinTransformer3Dv2(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (tuple[int]): Window size. Default: 4.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 # pretrain_img_size=224,
                 img_size=(512, 512, 32),
                 patch_size=(4, 4, 4),
                 in_chans=1, # 3
                 embed_dim=96,
                 depths=(3, 3, 6, 3),
                 num_heads=(3, 6, 12, 24),
                 window_size=((1, 7, 7), (1, 7, 7), (8, 1, 1)), # 7
                 dilate=((1, 1, 1), (1, 1, 1), (1, 1, 1)),
                 shift=((0, 0, 0), (0, 3, 3), (0, 0, 0)),
                 down_sample_size=(1, 2, 2),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        # self.pretrain_img_size = pretrain_img_size
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            # pretrain_img_size = to_2tuple(pretrain_img_size)
            # patch_size = to_2tuple(patch_size)
            # patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            if not isinstance(img_size, tuple):
                img_size = (img_size, img_size, img_size)
            if not isinstance(patch_size, tuple):
                patch_size = (patch_size, patch_size, patch_size)
            patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])

            # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dilate=dilate,
                shift=shift,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                down_sample_size=down_sample_size,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # input x: BxSxHxW
        x = torch.unsqueeze(x, 1)
        # unsqueeze x: Bx1xSxHxW
        x = self.patch_embed(x)

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Ws, Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Ws*Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2) # B Ws*Wh*Ww C
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                out = out.view(-1, self.num_features[i] * S, H, W).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3Dv2, self).train(mode)
        self._freeze_stages()
