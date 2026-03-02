"""Hyperbolic pooling layers for nnx."""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax.numpy as jnp
from flax import nnx
from flax import struct

from hypax.array import ManifoldArray
from hypax.manifolds import Manifold, PoincareBall
from hypax.manifolds.poincare_ball._linalg import poincare_unfold
from hypax.nn.helpers import tangent_space_fn


def _to_pair(value: int | Tuple[int, int], name: str) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must have length 2, got {value}")
        return tuple(int(v) for v in value)
    return (int(value), int(value))


def _compute_out_dim(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


@struct.dataclass
class _SpatialConfig:
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]


class HAvgPool2D(nnx.Module):
    """Hyperbolic average pooling based on Fréchet mean aggregation."""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        manifold: Manifold,
        *,
        stride: int | Tuple[int, int] | None = None,
        padding: int | Tuple[int, int] = 0,
        use_midpoint: bool = False,
    ):
        super().__init__()
        if not isinstance(manifold, PoincareBall):
            raise ValueError("HAvgPool2D currently supports the Poincaré ball manifold only")

        kernel = _to_pair(kernel_size, "kernel_size")
        stride = _to_pair(stride if stride is not None else kernel_size, "stride")
        pad = _to_pair(padding, "padding")

        self.config = nnx.static(_SpatialConfig(kernel_size=kernel, stride=stride, padding=pad))
        self.use_midpoint = nnx.static(use_midpoint)

    def __call__(self, x: ManifoldArray) -> ManifoldArray:
        if not isinstance(x, ManifoldArray):
            raise TypeError("HAvgPool2D expects a ManifoldArray input")

        batch_size, channels, height, width = x.shape
        kernel_h, kernel_w = self.config.kernel_size
        stride_h, stride_w = self.config.stride
        pad_h, pad_w = self.config.padding

        out_height = _compute_out_dim(height, kernel_h, stride_h, pad_h)
        out_width = _compute_out_dim(width, kernel_w, stride_w, pad_w)
        num_patches = out_height * out_width
        kernel_vol = kernel_h * kernel_w

        unfolded = poincare_unfold(
            x=x.data,
            kernel_size=self.config.kernel_size,
            in_channels=channels,
            c=x.manifold.curvature(),
            stride=self.config.stride,
            padding=self.config.padding,
            axis=1,
        )
        unfolded = unfolded.reshape(batch_size, channels, kernel_vol, num_patches)

        if self.use_midpoint:
            pooled = x.manifold.midpoint(
                unfolded,
                reduce_axis=2,
                axis=1,
            )
        else:
            pooled = x.manifold.frechet_mean(
                unfolded,
                reduce_axis=2,
                axis=1,
            )

        pooled = pooled.reshape(batch_size, channels, out_height, out_width)
        return x.replace(data=pooled)


class HMaxPool2D(nnx.Module):
    """Hyperbolic max pooling computed in the tangent space."""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        *,
        stride: int | Tuple[int, int] | None = None,
        padding: int | Tuple[int, int] = "Valid",
        dilation: int | Tuple[int, int] = 1,
    ):
        super().__init__()
        self.pool_fn = partial(nnx.max_pool, window_shape=kernel_size, strides=stride, padding=padding)

    def __call__(self, x: ManifoldArray) -> ManifoldArray:
        assert isinstance(x, ManifoldArray)

        x = tangent_space_fn(self.pool_fn)(x)
        return x