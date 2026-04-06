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
x_1 = \operatorname{Exp}_{x_0}(v)
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