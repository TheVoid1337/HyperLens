**HyperLens** is a one-step generative model based on Riemannian flow matching on product manifolds.  
The model predicts tangent vectors and maps them to the data space via geodesic shooting, removing the need for iterative sampling.

We define the latent space as a product manifold

$$
\mathcal{M} = \mathbb{E}^k \times \mathbb{H}^m \times \mathbb{S}^n
$$

with corresponding decomposition

$$
x = (x^{\mathbb{E}}, x^{\mathbb{H}}, x^{\mathbb{S}})
$$

and tangent space

$$
T_x \mathcal{M} = T_{x^{\mathbb{E}}}\mathbb{E}^k \oplus T_{x^{\mathbb{H}}}\mathbb{H}^m \oplus T_{x^{\mathbb{S}}}\mathbb{S}^n
$$

The model predicts a tangent vector

$$
v = f_\theta(x_0) = (v^{\mathbb{E}}, v^{\mathbb{H}}, v^{\mathbb{S}})
\in T_{x_0}\mathcal{M}
$$

Geodesic evolution is performed independently per component using the exponential map

$$
x_1 = \mathrm{Exp}_{x_0}(v)
= \left(
\operatorname{Exp}_{x^{\mathbb{E}}}(v^{\mathbb{E}}),
\operatorname{Exp}_{x^{\mathbb{H}}}(v^{\mathbb{H}}),
\operatorname{Exp}_{x^{\mathbb{S}}}(v^{\mathbb{S}})
\right)
$$

where each trajectory follows a geodesic $\gamma$ defined by the Levi-Civita connection

$$
\nabla_{\dot{\gamma}} \dot{\gamma} = 0, \quad
\gamma(0) = x_0, \quad
\dot{\gamma}(0) = v
$$

This yields a single-step mapping from the base distribution to the data manifold without numerical integration.

### Validation Results for VAE Reconstructions ImageNet-1K on different Image sizes and Training Phases and last 2 layers not frozen during training phase 2
| Training Phase | Image Size | FID Score | SSIM Score | PSNR Score |
|----------------|------------|-----------|------------|------------|
| Phase 1        | 256x256    | 2.71      | 0.85       | 30.20 dB   |
| Phase 1        | 512x512    | 0.56      | 0.87       | 31.57 dB   |
| Phase 2        | 256x256    | 11.84     | 0.86       | 30.55 dB   |
| Phase 2        | 512x512    | 0.83      | 0.88       | 32.25 dB   |
| Phase 3        | 256x256    | 5.84      | 0.86       | 30.34 dB   |
| Phase 3        | 512x512    | 0.62      | 0.87       | 31.87 dB   |


### Validation Results for VAE Reconstructions ImageNet-1K on different Image sizes and Training Phases and frozen latents during training phase 2
| Training Phase | Image Size | FID Score | SSIM Score | PSNR Score |
|----------------|------------|-----------|------------|------------|
| Phase 1        | 256x256    | 2.57      | 0.86       | 30.22 dB   |
| Phase 1        | 512x512    | 0.38      | 0.87       | 31.64 dB   |
| Phase 2        | 256x256    | 11.49     | 0.87       | 30.76 dB   |
| Phase 2        | 512x512    | 0.73      | 0.88       | 32.47 dB   |
| Phase 3        | 256x256    | 5.84      | 0.86       | 30.34 dB   |
| Phase 3        | 512x512    | 0.62      | 0.87       | 31.87 dB   |




### Validation Results for VAE Reconstructions ImageNet-1K
| Image Size | FID Score | SSIM Score | PSNR Score |
|------------|-----------|------------|------------|
| 256x256    | 6.48      | 0.8519     | 29.89 dB   |
| 512x512    | 0.75      | 0.8693     | 31.54 dB   |


