import jax
import jax.numpy as jnp
from flax import struct

from hypax.manifolds import Manifold

@struct.dataclass
class ManifoldArray:
    data: jax.Array
    manifold: Manifold
    axis: int

    @property
    def ndim(self):
        return self.data.ndim

    def dim(self) -> int:
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def flatten(self, manifold_axis, start_axis: int = 1, end_axis: int = -1):
        new_data = self.manifold.flatten(self.data, manifold_axis, start_axis=start_axis, end_axis=end_axis)
        return self.replace(data=new_data)
