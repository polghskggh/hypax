# parser.add_argument('--n-linear', type=int, default=2)
# hyperbolic_linear.py
#
# JAX/nnx implementation of a Poincaré fully-connected layer that mirrors
# `hypll.layers.HLinear` (PyTorch) while following the style of nnx.Linear.

from __future__ import annotations
import typing as tp

import chex
import jax.numpy as jnp
from flax import nnx

from hypax.array import ManifoldArray
from hypax.manifolds import Manifold


class HLinear(nnx.Module):
    """Hyperbolic (Poincaré) fully-connected layer for JAX/nnx."""

    def __init__(self, in_features: int, out_features: int, manifold: Manifold, use_bias: bool = True, *,
                 dtype: tp.Optional[jnp.dtype] = None, param_dtype: jnp.dtype = jnp.float32, rngs: nnx.Rngs):
        super().__init__()

        self.in_features = in_features
        weights, bias_value = manifold.construct_dl_parameters(
            in_features=in_features,
            out_features=out_features,
            bias=use_bias,
            key_z=rngs.params(),
            key_bias=rngs.params() if use_bias else None,
            dtype=param_dtype,
        )

        self.weights = nnx.Param(jnp.asarray(weights, dtype=param_dtype))
        self.bias = (
            nnx.Param(jnp.asarray(bias_value, dtype=param_dtype))
            if bias_value is not None
            else None
        )

    def __call__(self, x: ManifoldArray) -> ManifoldArray:
        """Apply the hyperbolic fully connected operation."""
        assert isinstance(x, ManifoldArray)
        chex.assert_shape(x.shape, (..., self.in_features))

        bias_value = self.bias.value if (self.bias is not None) else None
        result = x.manifold.fully_connected(
            x=x.data,
            z=self.weights.value,
            bias=bias_value,
            axis=-1,
        )
        return x.replace(data=result)
