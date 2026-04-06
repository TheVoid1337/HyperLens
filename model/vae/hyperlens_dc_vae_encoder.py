import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from dataclasses import dataclass

from model.abstractions import HyperLensDCVAEAbstractEncoder, HyperLensVAEModelParams
from model.layers import GLUBlock, ResidualCNNBlock, ResidualPixelUnshuffleDownsample, RMSNorm2d


@dataclass
class HyperLensDCVAEEncoderOutput:
    tangent_means: Tuple[Tensor, ...]
    tangent_log_vars: Tuple[Tensor, ...]


class HyperLensDCVAEEncoder(HyperLensDCVAEAbstractEncoder):
    def __init__(self, params: HyperLensVAEModelParams):
        super(HyperLensDCVAEEncoder, self).__init__(params)

        self.total_latent_dimension = sum(component.dimension for component in params.manifold_components)

        self.input_conv = nn.Conv2d(params.image_channels,
                                    params.hidden_channels, 3, stride=1, padding=1, bias=False)

        self.encoder = nn.ModuleList([])

        for i in range(params.number_of_layers):
            current_channels = params.channel_list[i]

            for _ in range(params.depth_list[i]):
                if current_channels >= 512:
                    self.encoder.append(GLUBlock(current_channels, params.head_dimension, params.number_of_heads))
                else:
                    self.encoder.append(ResidualCNNBlock(params.channel_list[i], params.channel_list[i]))

            if i < params.number_of_layers - 1:
                self.encoder.append(ResidualPixelUnshuffleDownsample(params.channel_list[i], params.channel_list[i+1]))


        self.activation = nn.SiLU()
        self.output_norm = RMSNorm2d(params.channel_list[-1])
        self.output_conv = nn.Conv2d(params.channel_list[-1], self.total_latent_dimension * 2, 1)


    def forward(self, image: Tensor) -> HyperLensDCVAEEncoderOutput:

        x = self.input_conv(image)

        for layer in self.encoder:
            x = layer(x)

        x = self.output_norm(x)
        x = self.activation(x)
        x = self.output_conv(x)

        total_tangent_mean, total_tangent_log_variance = torch.chunk(x, 2, dim=1)
        split_sizes = [c.dimension for c in self.params.manifold_components]
        split_means = torch.split(total_tangent_mean, split_sizes, dim=1)
        split_log_vars = torch.split(total_tangent_log_variance, split_sizes, dim=1)

        return HyperLensDCVAEEncoderOutput(
            tangent_means=split_means,
            tangent_log_vars=split_log_vars
        )