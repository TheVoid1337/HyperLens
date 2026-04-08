import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass


@dataclass
class ManifoldComponent:
    manifold_type: str # "hyperbolic", "euclidean", "spherical"
    dimension: int
    curvature: float = 1.0


class BaseManifoldWrapper:
    def inner_product(self, x:Tensor, u:Tensor, v:Tensor=None, keep_dim:bool=False) -> Tensor:
        raise NotImplementedError

    def exp_map(self, x:Tensor, u:Tensor) -> Tensor:
        raise NotImplementedError

    def log_map(self, x:Tensor, v:Tensor) -> Tensor:
        raise NotImplementedError

    def norm(self, x:Tensor, v:Tensor, keep_dim:bool=False) -> Tensor:
        raise NotImplementedError

    def project_to_u(self, x:Tensor, u:Tensor) -> Tensor:
        raise NotImplementedError

    def project_x(self, x:Tensor) -> Tensor:
        raise NotImplementedError

    def distance(self, x:Tensor, y:Tensor)-> Tensor:
        raise NotImplementedError


@dataclass
class HyperLensVAEModelParams:
    image_channels: int = 3
    hidden_channels: int = 32
    number_of_layers: int = 4
    number_of_heads: int = 8
    head_dimension: int = 16
    curvature : float = 1.0
    channel_list: list[int] = None
    depth_list: list[int] = None
    group_norm_channels: int = 16
    manifold_components: list[ManifoldComponent] = None


@dataclass
class HyperLensFlowTransformerModelParams:
    latent_channels: int = 256
    hidden_channels: int = 1024
    number_of_double_blocks: int = 12
    number_of_single_blocks: int = 24
    number_of_attention_heads: int = 16
    attention_head_dimension: int = 64
    text_embedding_dimension: int = 384
    time_dim: int = 100
    context_dim: int = 2048


class HyperLensAbstractModel(nn.Module):
    def __init__(self):
        super(HyperLensAbstractModel, self).__init__()


class HyperLensDCVAEAbstractModel(HyperLensAbstractModel):
    def __init__(self, params: HyperLensVAEModelParams):
        super(HyperLensDCVAEAbstractModel, self).__init__()
        self.params = params


class HyperLensFlowTransformerAbstractModel(HyperLensAbstractModel):
    def __init__(self, params: HyperLensFlowTransformerModelParams):
        super(HyperLensFlowTransformerAbstractModel, self).__init__()
        self.params = params


class HyperLensDCVAEAbstractEncoder(nn.Module):
    def __init__(self, params: HyperLensVAEModelParams):
        super(HyperLensDCVAEAbstractEncoder, self).__init__()
        self.params = params


class HyperLensDCVAEAbstractDecoder(nn.Module):
    def __init__(self, params: HyperLensVAEModelParams):
        super(HyperLensDCVAEAbstractDecoder, self).__init__()
        self.params = params

