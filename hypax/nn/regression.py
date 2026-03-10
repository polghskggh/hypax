import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import vmap

from hypax.array import ManifoldArray
from hypax.manifolds import Manifold

class HRegressionAngular(nnx.Module):
    """Hyperbolic (Poincaré) regression layer for JAX/nnx."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            manifold: Manifold,
            *,
            use_bias: bool = True,
            dtype: tp.Optional[jnp.dtype] = None,
            param_dtype: jnp.dtype = jnp.float32,
            l_bound: float = -1.0,
            u_bound: float = 1.0,
            rngs: nnx.Rngs,
    ):
        """
        Args
        ----
        in_features / out_features : Input & output feature sizes.
        manifold                  : Instance of `hypax.manifolds.Manifold`.
        use_bias                  : Attach bias term (default **True**).
        dtype                     : Computation dtype (default: infer).
        param_dtype               : Parameter dtype (default **float32**).
        rngs                      : `nnx.Rngs` container (use `.params()`).
        """
        super().__init__()

        self.in_features = in_features
        n_layers = out_features // 2

        z_keys = jax.random.split(rngs.params(), n_layers)
        b_keys = jax.random.split(rngs.params(), n_layers) if use_bias else None

        weights, bias_value = vmap(
            lambda kz, kb: manifold.construct_dl_parameters(
                in_features=in_features,
                out_features=2,
                bias=use_bias,
                key_z=kz,
                key_bias=kb,
                dtype=param_dtype,
            )
        )(z_keys, b_keys)
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.l_uncertainty = 0.001
        self.u_uncertainty = 0.5
        self.weights = nnx.Param(jnp.asarray(weights, dtype=param_dtype))
        self.bias = (
            nnx.Param(jnp.asarray(bias_value, dtype=param_dtype))
            if bias_value is not None
            else None
        )

    def __call__(self, x: ManifoldArray, eps=1e-7) -> ManifoldArray:
        """Apply the hyperbolic fully connected operation."""
        if not isinstance(x, ManifoldArray):
            raise TypeError(f"Input must be a ManifoldArray, got {type(x)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        bias_value = self.bias.value if (self.bias is not None) else None
        weight_value = self.weights.value


        independent_fc = vmap(lambda z, bias: x.manifold.fully_connected(x=x.data, z=z, bias=bias, axis=-1),
                              out_axes=1)


        raw_out = independent_fc(weight_value, bias_value)
        r2 = jnp.sum(raw_out ** 2, axis=-1, keepdims=True)
        angle = jnp.atan2(raw_out[..., 1], raw_out[..., 0])
        # angle = jnp.abs(angle) / jnp.pi * (self.u_bound - self.l_bound) + self.l_bound # range [-1 1] overflow at 0 # USED FOR CURRENT 30 DIM VERSION
        angle = angle / jnp.pi # range [-1, 1] -> overflow at -1 <-> 1  # USED FOR CURRENT 2 DIM VERSION
        # norm = (1 - norm.squeeze(-1) * jnp.sqrt(x.manifold.curvature())) * (self.u_uncertainty - self.l_uncertainty) + self.l_uncertainty
        norm = (1 - r2.squeeze(-1) * x.manifold.curvature()) * (self.u_uncertainty - self.l_uncertainty) + self.l_uncertainty
        # print(norm.shape)
        # norm = jnp.arctanh(1 - r2.squeeze(-1) * x.manifold.curvature() + 1e-5)
        result = jnp.concatenate((angle, norm), axis=-1)
        # print(result.shape)
        return result



        


