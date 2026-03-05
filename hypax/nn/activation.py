# activation.py
#
# JAX/nnx implementation of hyperbolic activation functions for the Poincaré ball model.

from __future__ import annotations

from flax import nnx

from hypax.nn.helpers import tangent_space_fn

hrelu = tangent_space_fn(nnx.relu)
