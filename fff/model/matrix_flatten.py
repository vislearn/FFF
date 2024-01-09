from math import sqrt, floor

from torch import nn

from fff.base import ModelHParams


class MatrixFlattenHParams(ModelHParams):
    reverse: bool = False


class MatrixFlatten(nn.Module):
    hparams: MatrixFlattenHParams

    def __init__(self, hparams: dict | MatrixFlattenHParams):
        if not isinstance(hparams, MatrixFlattenHParams):
            hparams = MatrixFlattenHParams(**hparams)
        super().__init__()
        self.hparams = hparams

    def encode(self, x, c):
        if self.hparams.reverse:
            return self._unflatten(x)
        return self._flatten(x)

    def decode(self, z, c):
        if self.hparams.reverse:
            return self._flatten(z)
        return self._unflatten(z)

    def _flatten(self, x):
        # Reshape from (..., n, n) to (..., n**2)
        return x.reshape(*x.shape[:-2], -1)

    def _unflatten(self, z):
        # Reshape from (..., n**2) to (..., n, n)
        dim = floor(sqrt(z.shape[-1]))
        assert dim ** 2 == z.shape[-1]
        return z.reshape(*z.shape[:-1], dim, dim)


class NonSquareMatrixFlattenHParams(MatrixFlattenHParams):
    reverse: bool = False
    original_shape: list[int]


class NonSquareMatrixFlatten(nn.Module):
    hparams = NonSquareMatrixFlattenHParams

    def __init__(self, hparams: dict | NonSquareMatrixFlattenHParams):
        if not isinstance(hparams, NonSquareMatrixFlattenHParams):
            hparams = NonSquareMatrixFlattenHParams(**hparams)
        super().__init__()
        self.hparams = hparams

    def encode(self, x, c):
        if self.hparams.reverse:
            return self._unflatten(x)
        return self._flatten(x)

    def decode(self, z, c):
        if self.hparams.reverse:
            return self._flatten(z)
        return self._unflatten(z)

    def _flatten(self, x):
        # Reshape from (..., n, m) to (..., n*m)
        return x.reshape(*x.shape[:-2], -1)

    def _unflatten(self, z):
        # Reshape from (..., n*m) to (..., n, m)
        return z.reshape(*z.shape[:-1], *self.hparams.original_shape)
