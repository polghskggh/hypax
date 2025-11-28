"""Unit tests for hyperbolic 2D convolution layer."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hypax.array import ManifoldArray
from hypax.nn import HConvolution2D

from src.hypax.manifolds.curvature import Curvature


class MockManifold(nnx.Module):
    """Mock manifold for testing."""

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.curvature = Curvature(c)



def test_hconv2d_initialization():
    """Test basic initialization of HConvolution2D."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        manifold=manifold,
        rngs=rngs,
    )

    # Check configuration
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3)
    assert conv.kernel_vol == 9
    assert conv.has_bias is True
    assert conv.stride == 1
    assert conv.padding == 0

    # Check parameter shapes
    assert conv.weights.value.shape == (
        3 * 9,
        16,
    )  # (in_channels * kernel_vol, out_channels)
    assert conv.bias.value.shape == (16,)


def test_hconv2d_tuple_kernel_size():
    """Test HConvolution2D with tuple kernel size."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 5),
        manifold=manifold,
        rngs=rngs,
    )

    assert conv.kernel_size == (3, 5)
    assert conv.kernel_vol == 15
    assert conv.weights.value.shape == (3 * 15, 8)


def test_hconv2d_no_bias():
    """Test HConvolution2D without bias."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        manifold=manifold,
        bias=False,
        rngs=rngs,
    )

    assert conv.has_bias is False
    assert conv.bias is None


def test_hconv2d_forward_pass_shape():
    """Test forward pass produces correct output shape."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=16,
        kernel_size=5,
        manifold=manifold,
        rngs=rngs,
    )

    # Create input: [batch=2, channels=3, height=32, width=32]
    data = jnp.ones((2, 3, 32, 32)) * 0.01  # Small values to stay in Poincare ball
    x = ManifoldArray(data=data, manifold=manifold)

    # Forward pass
    output = conv(x)

    # Check output is ManifoldArray
    assert isinstance(output, ManifoldArray)

    # Check output shape: [2, 16, 28, 28]
    # (32 - 5) / 1 + 1 = 28
    assert output.shape == (2, 16, 28, 28)


def test_hconv2d_with_stride():
    """Test HConvolution2D with stride > 1."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        manifold=manifold,
        stride=2,
        rngs=rngs,
    )

    # Create input: [batch=1, channels=3, height=32, width=32]
    data = jnp.ones((1, 3, 32, 32)) * 0.01
    x = ManifoldArray(data=data, manifold=manifold)

    # Forward pass
    output = conv(x)

    # Check output shape with stride=2: (32 - 3) / 2 + 1 = 15
    assert output.shape == (1, 16, 15, 15)


def test_hconv2d_with_padding():
    """Test HConvolution2D with padding."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        manifold=manifold,
        padding=1,
        rngs=rngs,
    )

    # Create input: [batch=1, channels=3, height=32, width=32]
    data = jnp.ones((1, 3, 32, 32)) * 0.01
    x = ManifoldArray(data=data, manifold=manifold)

    # Forward pass
    output = conv(x)

    # Check output shape with padding=1: (32 + 2*1 - 3) / 1 + 1 = 32
    # (same size as input when kernel_size=3, padding=1, stride=1)
    assert output.shape == (1, 16, 32, 32)


def test_hconv2d_with_stride_and_padding():
    """Test HConvolution2D with both stride and padding."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=8,
        kernel_size=5,
        manifold=manifold,
        stride=2,
        padding=2,
        rngs=rngs,
    )

    # Create input: [batch=1, channels=3, height=32, width=32]
    data = jnp.ones((1, 3, 32, 32)) * 0.01
    x = ManifoldArray(data=data, manifold=manifold)

    # Forward pass
    output = conv(x)

    # Check output shape: (32 + 2*2 - 5) / 2 + 1 = 16
    assert output.shape == (1, 8, 16, 16)


def test_hconv2d_identity_init():
    """Test identity initialization for square/tall weight matrices."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    # Create conv where in_features <= out_features for identity init
    conv = HConvolution2D(
        in_channels=2,
        out_channels=32,  # 2 * 9 = 18 < 32, so should use identity init
        kernel_size=3,
        manifold=manifold,
        id_init=True,
        rngs=rngs,
    )

    # Check that weights contain identity structure
    weights = conv.weights.value
    in_features = 2 * 9  # 18
    out_features = 32

    # First 18 rows of first 18 columns should approximate 0.5 * identity
    identity_block = weights[:in_features, :in_features]
    expected = 0.5 * jnp.eye(in_features)

    assert jnp.allclose(identity_block, expected)


def test_hconv2d_hnnpp_init():
    """Test HNN++ initialization for wide matrices."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    # Create conv where in_features > out_features or id_init=False
    conv = HConvolution2D(
        in_channels=8,
        out_channels=4,  # 8 * 9 = 72 > 4, so should use HNN++ init
        kernel_size=3,
        manifold=manifold,
        id_init=True,  # Will still use HNN++ because wide matrix
        rngs=rngs,
    )

    weights = conv.weights.value

    # Check that weights are not identity (use HNN++ initialization)
    in_features = 8 * 9
    out_features = 4

    # Verify shape
    assert weights.shape == (in_features, out_features)

    # Weights should have reasonable scale (not all zeros or too large)
    assert jnp.abs(weights).mean() > 0
    assert jnp.abs(weights).max() < 1.0

def test_hconv2d_gradient_flow():
    """Test that gradients can flow through the convolution."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        manifold=manifold,
        rngs=rngs,
    )

    # Create input
    data = jnp.ones((1, 3, 8, 8)) * 0.01
    x = ManifoldArray(data=data, manifold=manifold)

    # Define a simple loss function
    def loss_fn(conv):
        output = conv(x)
        return jnp.mean(output.data**2)

    # Compute gradients
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(conv)
    # Gradients should exist and be non-zero
    assert nnx.state(grads)['weights'].shape == conv.weights.value.shape
    assert not jnp.allclose(nnx.state(grads)['weights'], 0)


def test_hconv2d_invalid_input_type():
    """Test that HConvolution2D raises error for invalid input type."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        manifold=manifold,
        rngs=rngs,
    )

    # Try to pass raw array instead of ManifoldArray
    data = jnp.ones((1, 3, 8, 8))

    with pytest.raises(TypeError, match="Input must be a ManifoldArray"):
        conv(data)


def test_hconv2d_invalid_channels():
    """Test that HConvolution2D raises error for mismatched input channels."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        manifold=manifold,
        rngs=rngs,
    )

    # Create input with wrong number of channels
    data = jnp.ones((1, 5, 8, 8)) * 0.01  # 5 channels instead of 3
    x = ManifoldArray(data=data, manifold=manifold)

    with pytest.raises(ValueError, match="Expected 3 input channels"):
        conv(x)


def test_hconv2d_manifold_without_curvature():
    """Test that HConvolution2D raises error if manifold has no curvature."""

    class ManifoldNoCurvature(nnx.Module):
        """Manifold without curvature attribute."""
        pass

    manifold = ManifoldNoCurvature()
    rngs = nnx.Rngs(42)

    with pytest.raises(ValueError, match="Manifold must have"):
        HConvolution2D(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            manifold=manifold,
            rngs=rngs,
        )


def test_hconv2d_different_curvatures():
    """Test HConvolution2D with different curvature values."""
    for c_value in [0.5, 1.0, 2.0]:
        manifold = MockManifold(c=c_value)
        rngs = nnx.Rngs(42)

        conv = HConvolution2D(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            manifold=manifold,
            rngs=rngs,
        )

        # Create input
        data = jnp.ones((1, 3, 8, 8)) * 0.01
        x = ManifoldArray(data=data, manifold=manifold)

        # Forward pass should work
        output = conv(x)

        assert isinstance(output, ManifoldArray)
        assert output.shape == (1, 8, 6, 6)


def test_hconv2d_batch_processing():
    """Test HConvolution2D with different batch sizes."""
    manifold = MockManifold(c=1.0)
    rngs = nnx.Rngs(42)

    conv = HConvolution2D(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        manifold=manifold,
        rngs=rngs,
    )

    for batch_size in [1, 4, 8]:
        data = jnp.ones((batch_size, 3, 10, 10)) * 0.01
        x = ManifoldArray(data=data, manifold=manifold)

        output = conv(x)

        assert output.shape == (batch_size, 8, 8, 8)
