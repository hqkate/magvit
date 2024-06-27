import numpy as np
import mindspore as ms
from mindspore import ops

"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""


class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters,
        deterministic=False,
    ):
        self.parameters = parameters
        self.mean, self.logvar = ops.chunk(parameters, 2, axis=1)
        self.logvar = ops.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = ops.exp(0.5 * self.logvar)
        self.var = ops.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = ops.zeros_like(self.mean, dtype=self.mean.dtype)

    def sample(self):
        # torch.randn: standard normal distribution
        x = self.mean + self.std * ops.randn(self.mean.shape, dtype=self.mean.dtype)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return ms.Tensor([0.0])
        else:
            if other is None:  # SCH: assumes other is a standard normal distribution
                return 0.5 * ops.sum(
                    ops.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3, 4],
                )
            else:
                return 0.5 * ops.sum(
                    ops.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3, 4],
                )

    def nll(self, sample, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return ms.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * ops.sum(
            logtwopi + self.logvar + ops.pow(sample - self.mean, 2) / self.var, dim=dims
        )

    def mode(self):
        return self.mean
