from typing import Optional, Tuple, Type

from torch import Tensor
from dataclasses import dataclass

from model.abstractions import HyperLensDCVAEAbstractModel, HyperLensVAEModelParams
from model.vae.hyperlens_dc_vae_decoder import HyperLensDCVAEDecoder, HyperLensDCVAEDecoderOutput
from model.vae.hyperlens_dc_vae_encoder import HyperLensDCVAEEncoder
from model.vae.hyperlens_latent_sampler import HyperLensSamplerOutput, HyperLensSampler


@dataclass
class HyperLensDCVAEOutput:
    reconstructed_image: Tensor
    tangent_means: Optional[Type[Tuple[Tensor]]] = None
    tangent_log_vars: Optional[Type[Tuple[Tensor]]] = None


class HyperLensDCVAE(HyperLensDCVAEAbstractModel):
    def __init__(self, params: HyperLensVAEModelParams):
        super(HyperLensDCVAE, self).__init__(params)

        self.encoder = HyperLensDCVAEEncoder(params)

        self.sampler = HyperLensSampler(params)

        self.decoder = HyperLensDCVAEDecoder(params)



    def encode(self, image:Tensor) -> HyperLensSamplerOutput:

        encoded = self.encoder(image)

        z_sample = self.sampler(
            tangent_means=encoded.tangent_means,
            tangent_log_vars=encoded.tangent_log_vars
        )

        return z_sample


    def decode(self, latents: HyperLensSamplerOutput) -> HyperLensDCVAEDecoderOutput:
        return  self.decoder(latents.latent_vector_cat)


    def decode_from_latent(self, latent: Tensor) -> HyperLensDCVAEOutput:
        return HyperLensDCVAEOutput(reconstructed_image=self.decoder(latent).reconstructed_image)

    def forward(self, image:Tensor) -> HyperLensDCVAEOutput:
        encoded = self.encode(image)
        decoded = self.decode(encoded)
        return HyperLensDCVAEOutput(
            reconstructed_image=decoded.reconstructed_image,
            tangent_means=encoded.tangent_means,
            tangent_log_vars=encoded.tangent_log_vars
        )


