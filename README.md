# HyperLens

**A one-step generative model built on Riemannian flow matching over a product manifold of Euclidean, hyperbolic and spherical components.**

---

## What this is

Most latent generative models sample by integrating a trajectory: the model takes many small steps through the latent space before it produces an output. HyperLens tries to remove that loop. Instead of integrating, the network predicts a single tangent vector, and one exponential map carries the sample from the base distribution to the data manifold in a single shot.

The second idea is the geometry itself. Rather than a flat Euclidean latent space, HyperLens uses a **product manifold** $\mathcal{M} = \mathbb{E}^k \times \mathbb{H}^m \times \mathbb{S}^n$. The hyperbolic factor is well suited to hierarchical structure, the spherical factor to cyclic and directional structure, and the Euclidean factor covers everything else. The intent is a latent space whose curvature is a better match for the data than a single flat space would be.

**The goal in one sentence:** generate high-resolution images in a single step, from a latent space whose geometry reflects the structure of the data.

## Project status

This is an ongoing personal research project. It is not a finished model.

| Stage | Description | Status |
|---|---|---|
| 1 | Autoencoder with a product-manifold latent space, trained and evaluated on ImageNet-1K | **Done** — results below |
| 2 | Riemannian flow matcher on the frozen latent space | In progress |
| 3 | One-step generation via geodesic shooting, end-to-end evaluation | Planned |

Everything reported under *Results* refers to **Stage 1 only**. These are autoencoder reconstruction metrics, not generation metrics.

## Pipeline

1. The encoder maps an image to a point $x \in \mathcal{M}$ on the product manifold.
2. A network $f_\theta$ predicts a tangent vector at that point.
3. Geodesic shooting maps the tangent vector back to the manifold in one step.
4. The decoder reconstructs the image from the resulting latent point.

The autoencoder follows the design and the three-phase high-resolution training schedule of DC-AE [1], adapted here to a curved latent space. The manifold construction follows the mixed-curvature VAE [2], with the hyperbolic component built on the wrapped normal distribution [3] and the Poincaré ball formulation [4]. The flow matching objective follows Riemannian Flow Matching [5].

## Method

The latent space is a product of constant-curvature manifolds

$$
\mathcal{M} = \mathbb{E}^k \times \mathbb{H}^m \times \mathbb{S}^n
$$

so every latent point decomposes componentwise as

$$
x = (x^{\mathbb{E}},\ x^{\mathbb{H}},\ x^{\mathbb{S}})
$$

and the tangent space at $x$ decomposes as a direct sum

$$
T_x \mathcal{M} = T_{x^{\mathbb{E}}} \mathbb{E}^k \oplus T_{x^{\mathbb{H}}} \mathbb{H}^m \oplus T_{x^{\mathbb{S}}} \mathbb{S}^n
$$

The model predicts one tangent vector at the base point $x_0$

$$
v = f_\theta(x_0) = (v^{\mathbb{E}},\ v^{\mathbb{H}},\ v^{\mathbb{S}}) \in T_{x_0}\mathcal{M}
$$

Because the tangent space splits as a direct sum, the exponential map factorises and each component evolves independently

$$
x_1 = \mathrm{Exp}_{x_0}(v) = \big(\ \mathrm{Exp}_{x_0^{\mathbb{E}}}(v^{\mathbb{E}}),\ \ \mathrm{Exp}_{x_0^{\mathbb{H}}}(v^{\mathbb{H}}),\ \ \mathrm{Exp}_{x_0^{\mathbb{S}}}(v^{\mathbb{S}})\ \big)
$$

Each component follows a geodesic $\gamma$ of the Levi-Civita connection, that is, a curve with zero acceleration

$$
\nabla_{\dot{\gamma}} \dot{\gamma} = 0, \qquad \gamma(0) = x_0, \qquad \dot{\gamma}(0) = v
$$

Since all three factors have constant curvature, their exponential maps are available in closed form. The mapping from the base distribution to the data manifold therefore requires **no numerical integration** and no iterative solver.

## Results

Autoencoder reconstruction quality on ImageNet-1K, measured with FID, SSIM and PSNR. The three training phases are the decoupled high-resolution adaptation schedule from DC-AE [1]. The two tables differ only in how the latent space is treated during phase 2.

**Variant A — last two layers unfrozen during phase 2**

| Training Phase | Image Size | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| Phase 1 | 256×256 | 2.71 | 0.85 | 30.20 dB |
| Phase 1 | 512×512 | 0.56 | 0.87 | 31.57 dB |
| Phase 2 | 256×256 | 11.84 | 0.86 | 30.55 dB |
| Phase 2 | 512×512 | 0.83 | 0.88 | 32.25 dB |
| Phase 3 | 256×256 | 5.84 | 0.86 | 30.34 dB |
| Phase 3 | 512×512 | 0.62 | 0.87 | 31.87 dB |

**Variant B — latents frozen during phase 2**

| Training Phase | Image Size | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| Phase 1 | 256×256 | 2.57 | 0.86 | 30.22 dB |
| Phase 1 | 512×512 | 0.38 | 0.87 | 31.64 dB |
| Phase 2 | 256×256 | 11.49 | 0.87 | 30.76 dB |
| Phase 2 | 512×512 | 0.73 | 0.88 | 32.47 dB |
| Phase 3 | 256×256 | 5.84 | 0.86 | 30.34 dB |
| Phase 3 | 512×512 | 0.62 | 0.87 | 31.87 dB |

**Baseline**

| Image Size | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|
| 256×256 | 6.48 | 0.8519 | 29.89 dB |
| 512×512 | 0.75 | 0.8693 | 31.54 dB |

Two observations. Reconstruction FID is consistently better at 512×512 than at 256×256 across every phase and both variants. And phase 2 degrades 256×256 FID sharply in both variants while SSIM and PSNR stay flat or improve, which suggests the degradation is distributional rather than a loss of per-pixel fidelity.

## References

1. Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., Li, M., Lu, Y., Han, S. — *Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models.* ICLR 2025. [arXiv:2410.10733](https://arxiv.org/abs/2410.10733)
2. Skopek, O., Ganea, O.-E., Bécigneul, G. — *Mixed-curvature Variational Autoencoders.* ICLR 2020. [arXiv:1911.08411](https://arxiv.org/abs/1911.08411)
3. Nagano, Y., Yamaguchi, S., Fujita, Y., Koyama, M. — *A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning.* ICML 2019. [arXiv:1902.02992](https://arxiv.org/abs/1902.02992)
4. Mathieu, E., Le Lan, C., Maddison, C. J., Tomioka, R., Teh, Y. W. — *Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders.* NeurIPS 2019. [arXiv:1901.06033](https://arxiv.org/abs/1901.06033)
5. Chen, R. T. Q., Lipman, Y. — *Flow Matching on General Geometries.* ICLR 2024. [arXiv:2302.03660](https://arxiv.org/abs/2302.03660)
