import jax.numpy as jnp
import jax
from hypax.utils.math import beta_func


def rescale_norm(x, in_dims, out_dims):
    """Preserve the norm of reshaped vector.

    Apply beta-concatenation rescaling
    When concatenating vectors in hyperbolic space, we need to rescale to preserve norm
    beta_ni corresponds to the original dimension (in_channels)
    beta_n corresponds to the new dimension (in_channels * kernel_vol)
    """
    beta_ni = beta_func(in_dims / 2, 1 / 2)
    beta_n = beta_func(out_dims / 2, 1 / 2)
    rescale_factor = beta_n / beta_ni
    return x * rescale_factor

def safe_arccosh(x, eps=1e-15):
    x_square = (x - 1.0) * (x + 1.0)
    x_square = x_square + eps
    return jnp.log(x + jnp.sqrt(x_square))

def safe_norm(x, axis, keepdims=True, eps=1e-15):
    norm_squared = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
    norm_squared = norm_squared + eps
    norm = jnp.sqrt(norm_squared)
    return norm

def poincare_hyperplane_dists(x: jax.Array, z: jax.Array, r: jax.Array | None, c: jax.Array, axis: int = -1) -> jax.Array:
    """The Poincare signed distance to hyperplanes operation.

    Args:
        x (jax.Array): The input values.
        z (jax.Array): The hyperbolic vectors describing the hyperplane orientations
        r (jax.Array | None): The hyperplane offsets
        c (jax.Array): The curvature of the Poincare disk.
        axis (int, optional): The axis. Defaults to -1.

    Returns:
        jax.Array: signed distances of input w.r.t. the hyperplanes, denoted by v_k(x) in the HNN++ paper
    """
    axis_shifted_x = jnp.moveaxis(x, source=axis, destination=-1)

    c_sqrt = jnp.sqrt(c)
    lam = 2 / (1 - c * jnp.pow(axis_shifted_x, 2).sum(axis=-1, keepdims=True))
    z_norm = safe_norm(z, axis=0)
    if r is None:
        dim_shifted_output = (2 * z_norm / c_sqrt * jnp.asinh(c_sqrt * lam / z_norm * jnp.matmul(axis_shifted_x, z)))
    else:
        two_csqrt_r = 2.0 * c_sqrt * r
        dim_shifted_output = (2 * z_norm / c_sqrt * jnp.asinh(c_sqrt * lam / z_norm * jnp.matmul(axis_shifted_x, z)
                * jnp.cosh(two_csqrt_r)- (lam - 1) * jnp.sinh(two_csqrt_r)
            )
        )

    return jnp.moveaxis(dim_shifted_output, source=-1, destination=axis)
