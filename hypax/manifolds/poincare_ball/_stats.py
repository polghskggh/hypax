"""Statistical operations on the Poincaré ball (e.g., Fréchet mean/midpoint)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hypax.manifolds.poincare_ball._math import safe_arccosh, safe_norm

def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Axis {axis} is out of bounds for ndim={ndim}")
    return axis

def _align_axes_for_reduction(
        x: jax.Array,
        manifold_axis: int,
        reduce_axis: int,
) -> tuple[jax.Array, list[int], int]:
    ndim = x.ndim
    man = _normalize_axis(manifold_axis, ndim)
    red = _normalize_axis(reduce_axis, ndim)
    if man == red:
        raise ValueError("manifold_axis and reduce_axis must be different")

    axes = [i for i in range(ndim) if i not in (red, man)] + [red, man]
    permuted = jnp.transpose(x, axes)
    return permuted, axes, red

def _restore_axes(
    data: jax.Array,
    axes: list[int],
    reduce_axis: int,
    keepdims: bool,
) -> jax.Array:
    original_ndim = len(axes)
    current_order = axes[:-2] + [axes[-1]]
    target_order = [ax for ax in range(original_ndim) if ax != reduce_axis]
    perm = [current_order.index(ax) for ax in target_order]
    restored = jnp.transpose(data, perm)

    if keepdims:
        restored = jnp.expand_dims(restored, axis=reduce_axis)

    return restored

def _l_prime(y: jax.Array) -> jax.Array:
    cond = y < 1e-12
    safe_y = jnp.maximum(y, 1e-12)
    val = 2.0 * safe_arccosh(1 + 2.0 * safe_y) / jnp.sqrt(safe_y**2 + safe_y)
    return jnp.where(cond, 4.0, val)

def _frechet_ball_forward(
    X: jax.Array,
    c: jax.Array,
    *,
    max_iter: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> jax.Array:
    """Compute the Fréchet mean of ``X`` along its penultimate axis."""

    x_sq = jnp.sum(jnp.square(X), axis=-1)

    def body(carry, _):
        mu, converged = carry
        mu_sq = jnp.sum(jnp.square(mu), axis=-1, keepdims=False)
        diff = X - jnp.expand_dims(mu, axis=-2)
        xmu_sq = jnp.sum(jnp.square(diff), axis=-1)

        denom = (1 - c * x_sq) * (1 - c * jnp.expand_dims(mu_sq, axis=-1))
        denom = jnp.maximum(denom, 1e-15)
        ratio = c * xmu_sq / denom
        alphas = _l_prime(ratio) / jnp.maximum(1 - c * x_sq, 1e-15)

        c_term = jnp.sum(alphas * x_sq, axis=-1)
        b_term = jnp.sum(jnp.expand_dims(alphas, axis=-1) * X, axis=-2)
        a_term = jnp.sum(alphas, axis=-1)
        b_sq = jnp.sum(jnp.square(b_term), axis=-1)

        sqrt_term = jnp.sqrt(jnp.maximum((a_term + c * c_term) ** 2 - 4 * c * b_sq, 1e-15))
        denom_eta = jnp.maximum(2 * c * b_sq, 1e-15)
        eta = (a_term + c * c_term - sqrt_term) / denom_eta
        candidate = jnp.expand_dims(eta, axis=-1) * b_term

        # diff_norm = jnp.linalg.norm(candidate - mu, axis=-1)
        diff_norm = safe_norm(candidate - mu, axis=-1, keepdims=False)
        prev_norm = safe_norm(mu, axis=-1, keepdims=False)
        has_converged = (diff_norm < atol) | (diff_norm / prev_norm < rtol) | converged

        mu_next = jnp.where(jnp.expand_dims(has_converged, axis=-1), mu, candidate)
        return (mu_next, has_converged), None

    init_mu = X[..., 0, :]
    init_converged = jnp.zeros(init_mu.shape[:-1], dtype=bool)
    (final_mu, _), _ = jax.lax.scan(body, (init_mu, init_converged), xs=None, length=max_iter)
    return final_mu
