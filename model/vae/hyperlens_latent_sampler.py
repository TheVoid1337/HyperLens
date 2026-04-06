
import torch
import torch.nn as nn
from typing import Tuple, Type
from dataclasses import dataclass
from torch import Tensor

from model.abstractions import HyperLensVAEModelParams
from model.layers import HyperbolicPoincareManifold, SphericalManifold, EuclideanManifold


@dataclass
class HyperLensSamplerOutput:
    latent_vector_cat: Tensor
    tangent_means: Type[Tuple[Tensor, ...]] = None
    tangent_log_vars: Type[Tuple[Tensor, ...]] = None


class HyperLensSampler(nn.Module):
    def __init__(self, params: HyperLensVAEModelParams, sample_latent: bool = True):
        super(HyperLensSampler, self).__init__()
        self.params = params
        self.sample = sample_latent
        self.manifolds = []

        for comp in params.manifold_components:
            if comp.manifold_type == 'hyperbolic':
                self.manifolds.append(HyperbolicPoincareManifold(comp.curvature))
            elif comp.manifold_type == 'spherical':
                self.manifolds.append(SphericalManifold())
            elif comp.manifold_type == 'euclidean':
                self.manifolds.append(EuclideanManifold())
            else:
                raise ValueError(f"Unknown Manifold-Type: {comp.manifold_type}")

    @torch.compiler.disable
    def forward(self, tangent_means: Type[Tuple[Tensor, ...]],
                tangent_log_vars: Type[Tuple[Tensor, ...]]
                ) -> HyperLensSamplerOutput:

        latent_components = []

        for comp, manifold, mean, log_var in zip(self.params.manifold_components,
                                                 self.manifolds,
                                                 tangent_means,
                                                 tangent_log_vars):

            if comp.manifold_type == 'euclidean':
                if self.sample:
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(mean)
                    z = mean + std * eps
                else:
                    z = mean
                latent_components.append(z)

            elif comp.manifold_type == 'hyperbolic':

                with torch.autocast(device_type=mean.device.type, enabled=False):
                    mean_perm = mean.permute(0, 2, 3, 1).float()
                    log_variance_perm = log_var.permute(0, 2, 3, 1).float()

                    mean_curved = manifold.exp_map_0(mean_perm)

                    if self.sample:
                        std = torch.exp(0.5 * log_variance_perm)
                        eps_noise = std * torch.randn_like(mean_perm)
                        transported_noise = manifold.transport_0(mean_curved, eps_noise)
                        latents_curved_perm = manifold.exp_map(mean_curved, transported_noise)

                    else:
                        latents_curved_perm = mean_curved

                    latents = latents_curved_perm.permute(0, 3, 1, 2)

                    latent_components.append(latents)

            elif comp.manifold_type == 'spherical':
                with torch.autocast(device_type=mean.device.type, enabled=False):
                    mean_perm = mean.permute(0, 2, 3, 1).float()
                    log_variance_perm = log_var.permute(0, 2, 3, 1).float()
                    pole = torch.zeros_like(mean_perm)
                    pole[..., 0] = 1.0

                    v_mean = manifold.project_to_u(pole, mean_perm)

                    mean_curved = manifold.exp_map(pole, v_mean)

                    if self.sample:
                        std = torch.exp(0.5 * log_variance_perm)
                        raw_noise = torch.randn_like(mean_perm)
                        eps_noise = std * manifold.project_to_u(mean_curved, raw_noise)
                        latents_curved_perm = manifold.exp_map(pole, mean_curved + eps_noise)
                    else:
                        latents_curved_perm = mean_curved

                    latents = manifold.project_x(latents_curved_perm).permute(0, 3, 1, 2)
                    latent_components.append(latents)

        latents_cat = torch.cat(latent_components, dim=1)

        return HyperLensSamplerOutput(latent_vector_cat=latents_cat,
                                      tangent_means=tangent_means,
                                      tangent_log_vars=tangent_log_vars)



























