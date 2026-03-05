"""Unit tests for hyperbolic 2D convolution layer."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hypax.array import ManifoldArray
from hypax.nn import HConvolution2D
from hypax.manifolds.curvature import Curvature
from hypax.manifolds.poincare_ball import PoincareBall


def _manifold():
    return PoincareBall(Curvature(1.0, learnable=False))


def _nhwc(batch, height, width, channels, value=0.01):
    """Create small NHWC data inside the Poincaré ball."""
    return jnp.full((batch, height, width, channels), value)


# ── Initialisation ──────────────────────────────────────────────────────────

def test_hconv2d_basic_attributes():
    manifold = _manifold()
    conv = HConvolution2D(3, 16, 3, manifold, rngs=nnx.Rngs(0))

    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3)
    assert conv.kernel_vol == 9
    assert conv.has_bias is True
    assert conv.stride == (1, 1)
    assert conv.weights.value.shape == (3 * 9, 16)
    assert conv.bias.value.shape == (16,)


def test_hconv2d_tuple_kernel():
    manifold = _manifold()
    conv = HConvolution2D(3, 8, (3, 5), manifold, rngs=nnx.Rngs(0))
    assert conv.kernel_size == (3, 5)
    assert conv.kernel_vol == 15
    assert conv.weights.value.shape == (3 * 15, 8)


def test_hconv2d_no_bias():
    manifold = _manifold()
    conv = HConvolution2D(3, 16, 3, manifold, bias=False, rngs=nnx.Rngs(0))
    assert conv.has_bias is False
    assert conv.bias is None


# ── Forward pass (NHWC: batch, height, width, channels) ─────────────────────

def test_hconv2d_forward_shape_same_padding():
    """SAME padding: spatial dims are preserved."""
    manifold = _manifold()
    conv = HConvolution2D(3, 16, 5, manifold, rngs=nnx.Rngs(0))  # default "SAME"
    x = ManifoldArray(data=_nhwc(2, 32, 32, 3), manifold=manifold)
    out = conv(x)
    assert isinstance(out, ManifoldArray)
    assert out.shape == (2, 32, 32, 16)


def test_hconv2d_forward_shape_valid_padding():
    """VALID padding: spatial dims shrink by kernel_size - 1."""
    manifold = _manifold()
    conv = HConvolution2D(3, 16, 5, manifold, padding="VALID", rngs=nnx.Rngs(0))
    x = ManifoldArray(data=_nhwc(2, 32, 32, 3), manifold=manifold)
    out = conv(x)
    assert out.shape == (2, 28, 28, 16)


def test_hconv2d_stride_halves_spatial():
    """stride=2 with SAME padding halves spatial dims."""
    manifold = _manifold()
    conv = HConvolution2D(3, 16, 3, manifold, stride=2, rngs=nnx.Rngs(0))
    x = ManifoldArray(data=_nhwc(1, 32, 32, 3), manifold=manifold)
    out = conv(x)
    assert out.shape == (1, 16, 16, 16)


def test_hconv2d_batch_sizes():
    manifold = _manifold()
    conv = HConvolution2D(3, 8, 3, manifold, rngs=nnx.Rngs(0))
    for batch in [1, 4, 8]:
        x = ManifoldArray(data=_nhwc(batch, 10, 10, 3), manifold=manifold)
        out = conv(x)
        assert out.shape == (batch, 10, 10, 8)


def test_hconv2d_returns_manifold_array():
    manifold = _manifold()
    conv = HConvolution2D(3, 8, 3, manifold, rngs=nnx.Rngs(0))
    x = ManifoldArray(data=_nhwc(1, 8, 8, 3), manifold=manifold)
    out = conv(x)
    assert isinstance(out, ManifoldArray)
    assert out.manifold is manifold


def test_hconv2d_gradient_flow():
    manifold = _manifold()
    conv = HConvolution2D(3, 8, 3, manifold, rngs=nnx.Rngs(0))
    x = ManifoldArray(data=_nhwc(1, 8, 8, 3), manifold=manifold)

    def loss_fn(conv):
        return jnp.mean(conv(x).data ** 2)

    grads = jax.grad(loss_fn)(conv)
    assert nnx.state(grads)["weights"].shape == conv.weights.value.shape
    assert not jnp.allclose(nnx.state(grads)["weights"], 0)


# ── Error handling ───────────────────────────────────────────────────────────

def test_hconv2d_rejects_raw_array():
    manifold = _manifold()
    conv = HConvolution2D(3, 8, 3, manifold, rngs=nnx.Rngs(0))
    with pytest.raises(Exception):
        conv(jnp.ones((1, 8, 8, 3)))


def test_hconv2d_wrong_channels():
    manifold = _manifold()
    conv = HConvolution2D(3, 8, 3, manifold, rngs=nnx.Rngs(0))
    x = ManifoldArray(data=_nhwc(1, 8, 8, 5), manifold=manifold)  # 5 channels, not 3
    with pytest.raises(Exception):
        conv(x)


def test_hconv2d_manifold_without_curvature_raises():
    class NoCurvature(nnx.Module):
        pass

    with pytest.raises(Exception):
        HConvolution2D(3, 8, 3, NoCurvature(), rngs=nnx.Rngs(0))


# ── Initialisation schemes ───────────────────────────────────────────────────

def test_hconv2d_identity_init():
    manifold = _manifold()
    conv = HConvolution2D(2, 32, 3, manifold, id_init=True, rngs=nnx.Rngs(0))
    in_f = 2 * 9
    assert jnp.allclose(conv.weights.value[:in_f, :in_f], 0.5 * jnp.eye(in_f))


def test_hconv2d_hnnpp_init():
    manifold = _manifold()
    conv = HConvolution2D(8, 4, 3, manifold, id_init=True, rngs=nnx.Rngs(0))
    w = conv.weights.value
    assert w.shape == (8 * 9, 4)
    assert jnp.abs(w).mean() > 0
    assert jnp.abs(w).max() < 1.0


def test_hconv2d_different_curvatures():
    for c in [0.5, 1.0, 2.0]:
        manifold = PoincareBall(Curvature(c, learnable=False))
        conv = HConvolution2D(3, 8, 3, manifold, rngs=nnx.Rngs(0))
        x = ManifoldArray(data=_nhwc(1, 8, 8, 3), manifold=manifold)
        out = conv(x)
        assert isinstance(out, ManifoldArray)
        assert out.shape == (1, 8, 8, 8)
