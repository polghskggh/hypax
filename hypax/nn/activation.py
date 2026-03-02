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
from hypax.nn.helpers import tangent_space_fn

hrelu = tangent_space_fn(nnx.relu)
