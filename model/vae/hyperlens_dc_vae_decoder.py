import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

from model.abstractions import HyperLensVAEModelParams, HyperLensDCVAEAbstractDecoder
from model.layers import ResidualPixelShuffleUpsample, GLUBlock, ResidualCNNBlock, RMSNorm2d
from model.layers.manifold import EuclideanManifold, HyperbolicPoincareManifold, SphericalManifold


@dataclass
class HyperLensDCVAEDecoderOutput:
    reconstructed_image: Tensor


class HyperLensDCVAEDecoder(HyperLensDCVAEAbstractDecoder):
    def __init__(self, params: HyperLensVAEModelParams):
        super(HyperLensDCVAEDecoder, self).__init__(params)
        self.total_latent_dimension = sum(c.dimension for c in params.manifold_components)
        self.split_sizes = [c.dimension for c in params.manifold_components]
        self.manifolds = []
        for comp in params.manifold_components:
            if comp.manifold_type == 'hyperbolic':
                self.manifolds.append(HyperbolicPoincareManifold(comp.curvature))
            elif comp.manifold_type == 'spherical':
                self.manifolds.append(SphericalManifold())
            elif comp.manifold_type == 'euclidean':
                self.manifolds.append(EuclideanManifold())
            else:
                raise ValueError(f"Unknown manifold type: {comp.manifold_type}")


        channel_list = params.channel_list[::-1]
        depth_list = params.depth_list[::-1]

        self.pre_conv = nn.Conv2d(self.total_latent_dimension, channel_list[0], kernel_size=1)

        self.decoder_blocks = nn.ModuleList([])

        for i in range(params.number_of_layers):
            current_channels = channel_list[i]

            if i > 0:
                prev_channels = channel_list[i - 1]
                self.decoder_blocks.append(
                    ResidualPixelShuffleUpsample(prev_channels, current_channels)
                )

            for _ in range(depth_list[i]):
                if current_channels >= 512:
                    self.decoder_blocks.append(GLUBlock(current_channels, params.head_dimension))
                else:
                    self.decoder_blocks.append(ResidualCNNBlock(current_channels, current_channels))

        self.norm_out = RMSNorm2d(channel_list[-1])
        self.activation = nn.SiLU()
        self.output_conv = nn.Conv2d(channel_list[-1], params.image_channels, kernel_size=3, padding=1)


    @torch.compiler.disable
    def _map_from_manifold(self, latent_vector_cat: Tensor) -> Tensor:
        z_parts = torch.split(latent_vector_cat, self.split_sizes, dim=1)
        flat_z_components = []

        for comp, manifold, z_curved in zip(self.params.manifold_components, self.manifolds, z_parts):

            if comp.manifold_type == 'euclidean':
                flat_z_components.append(z_curved)

            elif comp.manifold_type == 'hyperbolic':
                with torch.autocast(device_type=z_curved.device.type, enabled=False):

                    z_p = z_curved.permute(0, 2, 3, 1).float()
                    z_flat_p = manifold.log_map_0(z_p)
                    z_flat = z_flat_p.permute(0, 3, 1, 2)
                    flat_z_components.append(z_flat)

            elif comp.manifold_type == 'spherical':

                with torch.autocast(device_type=z_curved.device.type, enabled=False):
                    z_p = z_curved.permute(0, 2, 3, 1).float()
                    pole = torch.zeros_like(z_p)
                    pole[..., 0] = 1.0
                    z_flat_p = manifold.log_map(pole, z_p)
                    z_flat = z_flat_p.permute(0, 3, 1, 2)
                    flat_z_components.append(z_flat)
            else:
                raise ValueError(f"Unknown manifold type: {comp.manifold_type}")

        return torch.cat(flat_z_components, dim=1)

    def forward(self, latent_vector_cat: Tensor) -> HyperLensDCVAEDecoderOutput:

        z_flat_concat = self._map_from_manifolds(latent_vector_cat)

        x = self.pre_conv(z_flat_concat)

        for layer in self.decoder_blocks:
            x = layer(x)

        x = self.norm_out(x)
        x = self.activation(x)
        x = self.output_conv(x)
        x = torch.tanh(x)
        return HyperLensDCVAEDecoderOutput(reconstructed_image=x)






