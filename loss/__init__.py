from .kl_divergence_loss import KLDivergenceLoss
from .dino_v2_perceptual_loss import DINOV2PerceptualLoss
from .latent_consistency_loss import LatentConsistencyLoss

__all__ = [
    "KLDivergenceLoss",
    "DINOV2PerceptualLoss",
    "LatentConsistencyLoss"
]
