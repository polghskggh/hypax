import jax
import jax.numpy as jnp
from flax import struct

from hypax.manifolds import Manifold

@struct.dataclass
class ManifoldArray:
    data: jax.Array
    manifold: Manifold

    @property
    def ndim(self):
        return self.data.ndim

    def dim(self) -> int:
        """PyTorch-style alias used by some shared helpers."""
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape
