import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.utils.checkpoint as checkpoint


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output

@HEADS.register_module()
class FCNHead_sp(FCNHead):
    def __init__(self,
                 sp_s=3,
                 num_sp=2,
                 sp_heads=8,
                 sp_channels=256,
                 **kwargs):
        super(FCNHead_sp, self).__init__(**kwargs)
        self.conv_seg = nn.Conv2d(self.channels * sp_s, self.num_classes, kernel_size=1)
        if num_sp == 0:
            self.sp_layers = nn.Identity()
        else:
            self.sp_layers = SpTr(num_sp=num_sp, sp_heads=sp_heads, sp_channels=sp_channels, out_channels=self.channels)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        _B, _S, _C, _H, _W = x.shape
        # x: (B, S, C, H, W)
        x = x.reshape((_B * _S, _C, _H, _W))
        # x: (B*S, C, H, W)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        # output: (B*S, C, H, W)
        output = output.reshape((_B, _S, -1, _H, _W))
        # output: (B, S, C, H, W)
        output = output.permute(1, 0, 3, 4, 2)
        # output: (S, B, H, W, C)
        output = output.reshape((_S, _B * _H * _W, -1))
        # output: (S, B*H*W, C)
        output = self.sp_layers(output) # apply spectral transformer
        # output: (S, B*H*W, C)
        output = output.reshape((_S, _B, _H, _W, -1))
        # output: (S, B, H, W, C)
        output = output.permute(1, 0, 4, 2, 3)
        # output: (B, S, C, H, W)
        output = output.reshape((_B, -1, _H, _W))
        # output: (B, S*C, H, W)
        output = self.cls_seg(output)
        return output

class SpTr(nn.Module):
    def __init__(self, num_sp, sp_heads, sp_channels, out_channels):
        super(SpTr, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(num_sp):
            self.layers.append(nn.TransformerEncoderLayer(out_channels, sp_heads, dim_feedforward=sp_channels))

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint.checkpoint(layer, x)
        return x



