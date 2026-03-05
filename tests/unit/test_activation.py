"""Unit tests for hyperbolic activation functions."""

import jax.numpy as jnp
from flax import nnx

from hypax.array import ManifoldArray
from hypax.nn import hrelu
from hypax.manifolds.curvature import Curvature
from hypax.manifolds.poincare_ball import PoincareBall


def _make_manifold(c=1.0):
    return PoincareBall(Curvature(c, learnable=False))


def test_hrelu_returns_manifold_array():
    manifold = _make_manifold()
    x = ManifoldArray(data=jnp.array([[0.1, -0.2], [0.3, -0.1]]), manifold=manifold)
    result = hrelu(x)
    assert isinstance(result, ManifoldArray)
    assert result.shape == x.shape


def test_hrelu_zeros():
    """Zero input (origin) should map back to origin."""
    manifold = _make_manifold()
    x = ManifoldArray(data=jnp.zeros((2, 3)), manifold=manifold)
    result = hrelu(x)
    assert jnp.allclose(result.data, jnp.zeros_like(x.data), atol=1e-6)


def test_hrelu_negative_values_suppressed():
    """Negative tangent components should be zeroed by relu."""
    manifold = _make_manifold()
    # Small negative values — logmap is approximately identity near origin
    data = jnp.array([[-0.05, -0.1]])
    x = ManifoldArray(data=data, manifold=manifold)
    result = hrelu(x)
    # Result should be near origin (relu kills negatives → expmap(0) = 0)
    assert jnp.all(jnp.abs(result.data) < jnp.abs(data))


def test_hrelu_preserves_positive():
    """Positive tangent components should remain non-zero."""
    manifold = _make_manifold()
    data = jnp.array([[0.1, 0.2]])
    x = ManifoldArray(data=data, manifold=manifold)
    result = hrelu(x)
    assert isinstance(result, ManifoldArray)
    # Result should be non-zero (positive values pass through relu)
    assert jnp.any(result.data != 0)


def test_hrelu_different_curvatures():
    for c in [0.5, 1.0, 2.0]:
        manifold = _make_manifold(c)
        x = ManifoldArray(data=jnp.array([[0.05, -0.05]]), manifold=manifold)
        result = hrelu(x)
        assert isinstance(result, ManifoldArray)
        assert result.shape == x.shape
