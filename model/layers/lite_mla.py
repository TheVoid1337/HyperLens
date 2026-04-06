import torch
from typing import Tuple

from torch import nn, Tensor
from torch.nn import functional as fun

class LiteMultiLayerAttention(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 num_heads: int,
                 head_dimension: int,
                 use_bias: bool = False,
                 eps: float = 1e-15,
                 scales: Tuple[int, ...] = (5,)
                 ):
        super(LiteMultiLayerAttention, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.eps = eps
        self.scales = scales
        self.heads = num_heads if num_heads is not None else int(input_channels // head_dimension)

        total_dimension = self.num_heads * head_dimension

        self.query_key_value_conv = nn.Conv2d(
            input_channels,
            3 * total_dimension,
            kernel_size=1,
            bias=use_bias
        )


        self.aggregation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * total_dimension,
                    3 * total_dimension,
                    kernel_size=scale,
                    padding=scale // 2,
                    groups=3 * total_dimension,
                    bias=use_bias,
                ),
                nn.Conv2d(
                    3 * total_dimension,
                    3 * total_dimension,
                    kernel_size=1,
                    groups=3 * self.heads,
                    bias=use_bias,
                )

            )
            for scale in scales
        ])


        self.kernel_function = nn.ReLU()
        self.projection_layer = (nn.Sequential(
            nn.Conv2d(total_dimension * (1 + len(scales)),
                                         output_channels,
                                         kernel_size=1,
                                         bias=use_bias
                                         ),
            nn.BatchNorm2d(output_channels)
        ))

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.compiler.disable
    def relu_linear_attention(self, qkv: Tensor) -> Tensor:
        b, _, h, w = qkv.shape

        if qkv.dtype in [torch.float16, torch.bfloat16]:
            qkv = qkv.float()

        qkv = qkv.view(b, -1, 3 * self.head_dimension, h * w)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q = self.kernel_function(q)
        k = self.kernel_function(k)
        trans_k = k.transpose(-1, -2)
        v = fun.pad(v, (0, 0, 0, 1), mode="constant", value=1)

        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)

        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(b, -1, h, w)


    @torch.autocast(device_type="cuda", enabled=False)
    @torch.compiler.disable
    def relu_quadratic_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        b, _, h, w = qkv.shape
        qkv = qkv.view(b, -1, 3 * self.head_dimension, h * w)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = self.kernel_function(q)
        k = self.kernel_function(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)

        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()

        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)

        return out.reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.query_key_value_conv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggregation:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        h, w = qkv.shape[-2:]
        if h * w > self.head_dimension:
            out = self.relu_linear_attention(qkv)
        else:
            out = self.relu_quadratic_attention(qkv)

        return self.projection_layer(out)