import geoopt
from torch import Tensor
from model.abstractions import BaseManifoldWrapper

class EuclideanManifold(BaseManifoldWrapper):
    def __init__(self):
        self.manifold = geoopt.Euclidean()

    def inner_product(self, x:Tensor, u:Tensor, v:Tensor=None, keep_dim:bool=False) -> Tensor:
        return self.manifold.inner(x, u, v, keepdim=keep_dim)

    def exp_map(self, x:Tensor, u:Tensor) -> Tensor:
        return self.manifold.expmap(x, u)

    def log_map(self, x:Tensor, v:Tensor) -> Tensor:
        return self.manifold.logmap(x, v)

    def norm(self, x:Tensor, v:Tensor, keep_dim:bool=False) -> Tensor:
        return self.manifold.norm(x, v, keepdim=keep_dim)

    def project_to_u(self, x:Tensor, u:Tensor) -> Tensor:
        return self.manifold.proju(x, u)

    def project_x(self, x:Tensor) -> Tensor:
        return self.manifold.projx(x)

    def distance(self, x:Tensor, y:Tensor)-> Tensor:
        return self.manifold.dist(x, y)



class SphericalManifold(BaseManifoldWrapper):
    def __init__(self):
        self.manifold = geoopt.Sphere()

    def inner_product(self, x:Tensor, u:Tensor, v:Tensor=None, keep_dim:bool=False) -> Tensor:
        return self.manifold.inner(x, u, v, keepdim=keep_dim)

    def exp_map(self, x:Tensor, u:Tensor) -> Tensor:
        return self.manifold.expmap(x, u)

    def log_map(self, x:Tensor, v:Tensor) -> Tensor:
        return self.manifold.logmap(x, v)

    def norm(self, x:Tensor, v:Tensor, keep_dim:bool=False) -> Tensor:
        return self.manifold.norm(x, v, keepdim=keep_dim)

    def project_to_u(self, x:Tensor, u:Tensor) -> Tensor:
        return self.manifold.proju(x, u)

    def project_x(self, x:Tensor) -> Tensor:
        return self.manifold.projx(x)

    def distance(self, x:Tensor, y:Tensor)-> Tensor:
        return self.manifold.dist(x, y)


class HyperbolicPoincareManifold(BaseManifoldWrapper):
    def __init__(self, curvature: float = 1.0):
        """
        Initialize the Hyperbolic Poincare Manifold.
        :param curvature: The curvature of the manifold is set to 1.0 by default.
        Geoopt uses the exact formulation of the PoincareBallExact manifold.
        The curvature is negative.
        """
        self.manifold = geoopt.PoincareBallExact(c=curvature)

    def inner_product(self, x:Tensor, u:Tensor, v:Tensor=None, keep_dim:bool=False) -> Tensor:
        return self.manifold.inner(x, u, v, keepdim=keep_dim)

    def exp_map(self, x:Tensor, u:Tensor) -> Tensor:
        return self.manifold.expmap(x, u)

    def exp_map_0(self, x:Tensor) -> Tensor:
        return self.manifold.expmap0(x)

    def log_map(self, x:Tensor, v:Tensor) -> Tensor:
        return self.manifold.logmap(x, v)

    def log_map_0(self, x:Tensor) -> Tensor:
        return self.manifold.logmap0(x)

    def norm(self, x:Tensor, v:Tensor, keep_dim:bool=False) -> Tensor:
        return self.manifold.norm(x, v, keepdim=keep_dim)

    def project_to_u(self, x:Tensor, u:Tensor) -> Tensor:
        return self.manifold.proju(x, u)

    def project_x(self, x:Tensor) -> Tensor:
        return self.manifold.projx(x)

    def distance(self, x:Tensor, y:Tensor)-> Tensor:
        return self.manifold.dist(x, y)

    def transport_0(self, x:Tensor, u:Tensor) -> Tensor:
        return self.manifold.transp0(x, u)