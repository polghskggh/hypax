import jax.numpy as jnp
import jax
from hypax.utils.math import beta_func

def mobius_add(x: jax.Array, y: jax.Array, c: jax.Array, axis: int = -1) -> jax.Array:
    broadcast_dim = max(x.ndim, y.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    x2 = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    y2 = jnp.square(y).sum(axis=axis - broadcast_dim + y.ndim, keepdims=True)

    xy = (x * y).sum(axis=axis, keepdims=True)

    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = jnp.maximum(1 + 2 * c * xy + c**2 * x2 * y2, 1e-15)

    return numerator / denominator
def project(x: jax.Array, c: jax.Array, axis: int = -1, eps: float = -1.0) -> jax.Array:
    eps = jnp.asarray(eps, dtype=x.dtype)
    eps_val = jnp.where(
        eps < 0, jnp.array(4e-3 if x.dtype == jnp.float32 else 1e-5, dtype=x.dtype), eps
    )

    maxnorm = (1 - eps_val) / jnp.sqrt(c + 1e-15)
    maxnorm = jnp.where(c > 0, maxnorm, jnp.full_like(c, 1e15))
    norm = safe_norm(x, axis)

    cond = norm > maxnorm
    projected = x / norm * maxnorm

    return jnp.where(cond, projected, x)


def expmap0(v: jax.Array, c: jax.Array, axis: int = -1):
    v_norm = safe_norm(v, axis)
    v_norm_c_sqrt = v_norm * jnp.sqrt(c)
    return project(jnp.tanh(v_norm_c_sqrt) * v / v_norm_c_sqrt, c, axis=axis)


def logmap0(y: jax.Array, c: jax.Array, axis: int = -1):
    y_norm = safe_norm(y, axis)
    y_norm_c_sqrt = y_norm * jnp.sqrt(c)
    return jnp.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt


def expmap(x: jax.Array, v: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(x.ndim, v.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    v_norm = safe_norm(v, axis=axis - broadcast_dim + v.ndim)

    x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)

    c_sqrt = jnp.sqrt(c)
    second_term = jnp.tanh(c_sqrt * lambda_x * v_norm / 2) * v / (c_sqrt * v_norm)

    return project(mobius_add(x, second_term, c, axis=axis), c, axis=axis)


def logmap(x: jax.Array, y: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(x.ndim, y.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    min_x_y = mobius_add(-x, y, c, axis=axis)
    min_x_y_norm = safe_norm(min_x_y, axis=axis)

    x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)

    c_sqrt = jnp.sqrt(c)
    return (
        2
        / (c_sqrt * lambda_x)
        * jnp.atanh(c_sqrt * min_x_y_norm)
        * min_x_y
        / min_x_y_norm
    )


def gyration(u: jax.Array, v: jax.Array, w: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(u.ndim, v.ndim, w.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis
    u2 = jnp.sum(u**2, axis=axis - broadcast_dim + u.ndim, keepdims=True)
    v2 = jnp.sum(v**2, axis=axis - broadcast_dim + v.ndim, keepdims=True)
    uv = jnp.sum(u * v, axis=axis - broadcast_dim + max(u.ndim, v.ndim), keepdims=True)
    uw = jnp.sum(u * w, axis=axis - broadcast_dim + max(u.ndim, w.ndim), keepdims=True)
    vw = jnp.sum(v * w, axis=axis - broadcast_dim + max(v.ndim, w.ndim), keepdims=True)

    K2 = c**2
    a = -K2 * uw * v2 + c * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + K2 * u2 * v2

    return w + 2 * (a * u + b * v) / jnp.maximum(d, 1e-15)


def mobius_add_batch(x: jax.Array, y: jax.Array, c: jax.Array) -> jax.Array:
    xy = jnp.einsum("bij,bkj->bik", x, y)
    x2 = jnp.square(x).sum(axis=-1, keepdims=True)
    y2 = jnp.square(y).sum(axis=-1, keepdims=True)
    num = 1 + 2 * c * xy + c * jnp.permute_dims(y2, (0, 2, 1))
    num = jnp.expand_dims(num, axis=2) * jnp.expand_dims(x, axis=2)
    num = num + jnp.expand_dims(1 - c * x2, axis=3) * jnp.expand_dims(y, axis=1)
    denom = 1 + 2 * c * xy + jnp.square(c) * x2 * jnp.permute_dims(y2, (0, 2, 1))
    return num / jnp.clip(jnp.expand_dims(denom, axis=3), min=1e-15)

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
