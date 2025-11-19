"""Tests for hyperbolic pooling layers."""

import jax.numpy as jnp
from flax import nnx

from hypax.array import ManifoldArray
from hypax.manifolds.poincare_ball import PoincareBall
from hypax.nn import HAvgPool2D, HMaxPool2D


def _make_input(batch=1, channels=4, height=8, width=8, value=0.05):
    data = jnp.full((batch, channels, height, width), value)
    manifold = PoincareBall(c=1.0)
    return ManifoldArray(data=data, manifold=manifold), manifold


def test_havgpool2d_basic_shape():
    """HAvgPool2D should preserve manifold data and shrink spatial dims."""
    x, manifold = _make_input()
    pool = HAvgPool2D(kernel_size=2, manifold=manifold, stride=2)

    out = pool(x)

    assert isinstance(out, ManifoldArray)
    assert out.manifold is manifold
    assert out.shape == (1, 4, 4, 4)


def test_havgpool2d_midpoint_matches_frechet_on_constant_input():
    """Midpoint and Fréchet pooling should be identical on constant tensors."""
    x, manifold = _make_input(value=0.02)
    pool_mean = HAvgPool2D(kernel_size=2, manifold=manifold, stride=2, use_midpoint=False)
    pool_mid = HAvgPool2D(kernel_size=2, manifold=manifold, stride=2, use_midpoint=True)

    out_mean = pool_mean(x)
    out_mid = pool_mid(x)

    assert jnp.allclose(out_mean.data, out_mid.data, atol=1e-6)


def test_hmaxpool2d_matches_tangent_space_pooling():
    """HMaxPool2D should match the op_in_tangent_space construction."""
    x, manifold = _make_input(height=6, width=6, value=0.01)
    pool = HMaxPool2D(kernel_size=2, manifold=manifold, stride=2)
    out = pool(x)

    tangent = manifold.logmap(y=x.data, x=None, axis=1)
    nhwc = jnp.transpose(tangent, (0, 2, 3, 1))
    ref = nnx.max_pool(nhwc, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    ref_nchw = jnp.transpose(ref, (0, 3, 1, 2))
    expected = manifold.expmap(v=ref_nchw, x=None, axis=1)

    assert jnp.allclose(out.data, expected)


def test_pooling_rejects_raw_arrays():
    """Pooling ops should require ManifoldArray inputs."""
    _, manifold = _make_input()
    pool = HAvgPool2D(kernel_size=2, manifold=manifold)

    raw = jnp.ones((1, 4, 8, 8)) * 0.01

    try:
        pool(raw)  # type: ignore[arg-type]
    except TypeError:
        return

    raise AssertionError("Expected TypeError when passing raw array to HAvgPool2D")
