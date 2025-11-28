"""Unit tests for hyperbolic activation functions."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hypax.array import ManifoldArray
from hypax.nn import HReLU, hrelu

from hypax.manifolds.curvature import Curvature


class MockManifold(nnx.Module):
    """Mock manifold for testing."""

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.curvature = Curvature(c)


def test_hrelu_function_basic():
    """Test basic hrelu function."""
    # Create a simple manifold
    manifold = MockManifold(c=1.0)

    # Create test data with positive and negative components
    data = jnp.array([[0.1, -0.2], [0.3, -0.1]])
    x = ManifoldArray(data=data, manifold=manifold)

    # Apply hrelu
    result = hrelu(x, c=manifold.curvature())

    # Check that result is a ManifoldArray
    assert isinstance(result, ManifoldArray)

    # Check that result has same shape
    assert result.shape == data.shape

    # Check that result data is different from input (ReLU should have effect)
    assert not jnp.allclose(result.data, data)


def test_hrelu_module():
    """Test HReLU module."""
    # Create a simple manifold
    manifold = MockManifold(c=1.0)

    # Create activation module
    activation = HReLU()

    # Create test data
    data = jnp.array([[0.2, -0.3], [-0.1, 0.4]])
    x = ManifoldArray(data=data, manifold=manifold)

    # Apply activation
    result = activation(x)

    # Check that result is a ManifoldArray
    assert isinstance(result, ManifoldArray)

    # Check that result has same shape
    assert result.shape == data.shape


def test_hrelu_zeros():
    """Test hrelu with zero input."""
    manifold = MockManifold(c=1.0)

    # Zero input should map to zero (origin of manifold)
    data = jnp.zeros((2, 3))
    x = ManifoldArray(data=data, manifold=manifold)

    result = hrelu(x, c=manifold.curvature())

    # Result should also be zero
    assert jnp.allclose(result.data, jnp.zeros_like(data), atol=1e-6)


def test_hrelu_positive_values():
    """Test hrelu with all positive values in tangent space."""
    manifold = MockManifold(c=1.0)

    # Small positive values that should remain largely unchanged
    data = jnp.array([[0.1, 0.2], [0.15, 0.05]])
    x = ManifoldArray(data=data, manifold=manifold)

    result = hrelu(x, c=manifold.curvature())

    # Check that result is valid
    assert isinstance(result, ManifoldArray)
    assert result.shape == data.shape


def test_hrelu_custom_axis():
    """Test hrelu with custom axis."""
    manifold = MockManifold(c=1.0)

    # Create HReLU with custom axis
    activation = HReLU(axis=-2)

    data = jnp.array([[0.1, -0.2], [0.3, -0.1]])
    x = ManifoldArray(data=data, manifold=manifold)

    result = activation(x)

    # Check that result is valid
    assert isinstance(result, ManifoldArray)
    assert result.shape == data.shape


def test_hrelu_curvature_extraction():
    """Test that hrelu can extract curvature from manifold."""
    manifold = MockManifold(c=2.0)

    data = jnp.array([[0.05, -0.1]])
    x = ManifoldArray(data=data, manifold=manifold)

    # Call without explicit curvature - should extract from manifold
    result = hrelu(x)

    assert isinstance(result, ManifoldArray)
    assert result.shape == data.shape


def test_hrelu_missing_curvature_raises():
    """Test that hrelu raises error if curvature not available."""

    class ManifoldNoCurvature(nnx.Module):
        """Manifold without curvature attribute."""

        pass

    manifold = ManifoldNoCurvature()
    data = jnp.array([[0.1, 0.2]])
    x = ManifoldArray(data=data, manifold=manifold)

    # Should raise ValueError when curvature cannot be extracted
    with pytest.raises(ValueError, match="Curvature not provided"):
        hrelu(x)
