import torch
import numpy as np
from torch.nn import Embedding


class EuclideanManifold():
    __slots__ = ["max_norm"]

    def __init__(self, max_norm=None, K=None, **kwargs):
        self.max_norm = max_norm
        if K is not None:
            self.K = K
            self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))

    # def allocate_lt(self, N, dim, sparse):
    #     return Embedding(N, dim, sparse=sparse)
    
    # def init_weights(self, w, scale=1e-4):
    #     w.weight.data.uniform_(-scale, scale)

    def norm(self, u, **kwargs):
        if isinstance(u, Embedding):
            u = u.weight
        return u.pow(2).sum(dim=-1).sqrt()

    def normalize(self, u):
        d = u.size(-1)
        if self.max_norm:
            u.view(-1, d).renorm_(2, 0, self.max_norm)
        return u

    def distance(self, u, v):
        return (u - v).pow(2).sum(dim=-1)

    def rgrad(self, p, d_p):
        return d_p

    def half_aperture(self, u):
        sqnu = u.pow(2).sum(dim=-1)
        return torch.asin(self.inner_radius / sqnu.sqrt())

    def angle_at_u(self, u, v):
        norm_u = self.norm(u)
        norm_v = self.norm(v)
        dist = self.distance(v, u)
        num = norm_u.pow(2) - norm_v.pow(2) - dist.pow(2)
        denom = 2 * norm_v * dist
        return (num / denom).acos()

    def expm(self, p, d_p, normalize=False, lr=None, out=None):
        if lr is not None:
            d_p.mul_(-lr)
        if out is None:
            out = p
        out.add_(d_p)
        if normalize:
            self.normalize(out)
        return out

    def logm(self, p, d_p, out=None):
        return p - d_p

    def ptransp(self, p, x, y, v):
        ix, v_ = v._indices().squeeze(), v._values()
        return p.index_copy_(0, ix, v_)