import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as fun

from model.layers.lite_mla import LiteMultiLayerAttention


class ResidualPixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualPixelShuffleUpsample, self).__init__()
        self.expand_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.act = nn.SiLU()
        self.smooth_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = fun.pixel_shuffle(x, upscale_factor=2)

        b, c_shortcut, h, w = shortcut.shape
        repeats = self.expand_conv.out_channels // 4 // c_shortcut
        shortcut = shortcut.repeat(1, repeats, 1, 1)

        x_learned = self.expand_conv(x)
        x_learned = self.pixel_shuffle(x_learned)
        x_learned = self.act(x_learned)
        x_learned = self.smooth_conv(x_learned)

        return x_learned + shortcut


class ResidualPixelUnshuffleDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualPixelUnshuffleDownsample, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.compress_conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:

        shortcut = fun.pixel_unshuffle(x, downscale_factor=2)

        b, c_shortcut, h, w = shortcut.shape
        out_c = self.compress_conv.out_channels
        groups = c_shortcut // out_c

        shortcut = shortcut.view(b, groups, out_c, h, w).mean(dim=1)


        x_learned = self.pixel_unshuffle(x)
        x_learned = self.compress_conv(x_learned)
        x_learned = self.act(x_learned)

        return x_learned + shortcut


class ResidualCNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualCNNBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.group_norm_1 = nn.GroupNorm(16, input_channels, eps=1e-6, affine=True)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1)
        self.group_norm_2 = nn.GroupNorm(16, output_channels,eps=1e-6, affine=True)
        self.activation = nn.SiLU()
        self.res_con = nn.Conv2d(input_channels, output_channels,
                                 1, stride=1, padding=0) \
            if input_channels != output_channels else nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        x = x
        y = self.group_norm_1(x)
        y = self.activation(y)
        y = self.conv_1(y)
        y = self.group_norm_2(y)
        y = self.activation(y)
        y = self.conv_2(y)
        return y + self.res_con(x)


class GLUMBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            expand_ratio: float = 6.0,
            use_bias: bool = False,
    ):
        super(GLUMBConv, self).__init__()
        mid_channels = int(round(in_channels * expand_ratio))

        self.glu_act = nn.SiLU()

        self.inverted_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * 2, kernel_size=1, bias=use_bias),
            nn.SiLU()
        )

        self.depth_conv = nn.Conv2d(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=mid_channels * 2,
            bias=use_bias
        )

        self.point_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=use_bias),
            nn.GroupNorm(1, out_channels, eps=1e-5)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x


class GLUBlock(nn.Module):
    def __init__(self, channels:int, dimension:int, num_heads:int = 8):
        super(GLUBlock, self).__init__()

        self.norm_1 = RMSNorm2d(channels)

        self.context_module = LiteMultiLayerAttention(input_channels=channels,
                                                      output_channels=channels,
                                                      num_heads=num_heads,
                                                      head_dimension=dimension
                                                      )

        self.norm_2 = RMSNorm2d(channels)

        self.local_module = GLUMBConv(in_channels=channels, out_channels=channels)


    def forward(self, x: Tensor) -> Tensor:
        x = x + self.context_module(self.norm_1(x))
        x = x + self.local_module(self.norm_2(x))
        return x


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super(RMSNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(x.dtype)