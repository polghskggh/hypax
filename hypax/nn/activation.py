# activation.py
#
# JAX/nnx implementation of hyperbolic activation functions for the Poincaré ball model.

from __future__ import annotations

from flax import nnx

from hypax.array import ManifoldArray
from hypax.manifolds.poincare_ball._math import safe_norm
from hypax.nn.helpers import tangent_space_fn

def htanh(x: ManifoldArray) -> ManifoldArray:
    norm = safe_norm(x.data, axis=x.axis)
    result = nnx.tanh(norm) * x.data / norm
    return x.replace(data=result)

hrelu = tangent_space_fn(nnx.relu)
