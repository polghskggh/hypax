# activation.py
#
# JAX/nnx implementation of hyperbolic activation functions for the Poincaré ball model.

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
from flax import nnx
from hypax.array import ManifoldArray
from hypax.manifolds.poincare_ball._diffgeom import logmap0, expmap0

def tangent_space_fn(x: ManifoldArray, tangent_fun: Callable[[jax.Array], jax.Array]) -> ManifoldArray:
    data = x.data
    manifold = x.manifold
    c = manifold.curvature()
    tangent = logmap0(data, c, axis=x.axis)
    tangent_relu = tangent_fun(tangent)
    result = expmap0(tangent_relu, c, axis=x.axis)
    return x.replace(data=result)

hrelu = partial(tangent_space_fn, tangent_fun=nnx.relu)
