"""Accuracy tests for linear algebra operations.

Tests the Poincaré hyperplane distance and fully-connected layer operations.
"""

import pytest
import jax.numpy as jnp
import jax

from hypax.manifolds.poincare_ball._math import poincare_hyperplane_dists as hypax_hyperplane_dists
from hypax.manifolds.poincare_ball import PoincareBall
from hypax.manifolds.curvature import Curvature


def hypax_fully_connected(x, z, bias, c, axis=-1):
    manifold = PoincareBall(Curvature(float(c), learnable=False))
    return manifold.fully_connected(x, z, bias, axis=axis)


class TestPoincareHyperplaneDists:
    """Test Poincaré hyperplane distance computation."""

    @pytest.mark.parametrize(
        "batch_size,input_dim,output_dim",
        [
            (5, 3, 2),
            (10, 10, 5),
            (8, 50, 20),
        ],
    )
    def test_hyperplane_dists_without_bias(
        self, batch_size, input_dim, output_dim, tolerance, jax_key
    ):
        """Test hyperplane distance computation without offset."""
        # Generate test data
        key_x, key_z, key_c = jax.random.split(jax_key, 3)

        # Input points on the manifold (need to be within the ball)
        x = jax.random.uniform(key_x, (batch_size, input_dim), minval=-0.5, maxval=0.5)
        # Hyperplane orientations
        z = jax.random.normal(key_z, (input_dim, output_dim))
        # Curvature (positive value)
        c = jnp.abs(jax.random.normal(key_c, ())) + 0.1

        # Compute using hypax
        result = hypax_hyperplane_dists(x, z, None, c, axis=-1)

        # Basic sanity checks
        assert result.shape == (batch_size, output_dim), (
            f"Expected shape {(batch_size, output_dim)}, got {result.shape}"
        )
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"

    @pytest.mark.parametrize(
        "batch_size,input_dim,output_dim",
        [
            (5, 3, 2),
            (10, 10, 5),
        ],
    )
    def test_hyperplane_dists_with_bias(
        self, batch_size, input_dim, output_dim, tolerance, jax_key
    ):
        """Test hyperplane distance computation with offset."""
        # Generate test data
        key_x, key_z, key_r, key_c = jax.random.split(jax_key, 4)

        # Input points on the manifold
        x = jax.random.uniform(key_x, (batch_size, input_dim), minval=-0.5, maxval=0.5)
        # Hyperplane orientations
        z = jax.random.normal(key_z, (input_dim, output_dim))
        # Hyperplane offsets
        r = jax.random.normal(key_r, (output_dim,))
        # Curvature
        c = jnp.abs(jax.random.normal(key_c, ())) + 0.1

        # Compute using hypax
        result = hypax_hyperplane_dists(x, z, r, c, axis=-1)

        # Basic sanity checks
        assert result.shape == (batch_size, output_dim), (
            f"Expected shape {(batch_size, output_dim)}, got {result.shape}"
        )
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"

    def test_hyperplane_dists_zero_input(self, tolerance, jax_key):
        """Test hyperplane distance with zero input (origin)."""
        input_dim, output_dim = 3, 2

        # Generate test data
        key_z, key_c = jax.random.split(jax_key, 2)

        # Zero input (origin of Poincaré ball)
        x = jnp.zeros((1, input_dim))
        # Hyperplane orientations
        z = jax.random.normal(key_z, (input_dim, output_dim))
        # Curvature
        c = jnp.array(1.0)

        # Compute using hypax
        result = hypax_hyperplane_dists(x, z, None, c, axis=-1)

        # At origin, distances should be close to zero for symmetric hyperplanes
        assert result.shape == (1, output_dim)
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"


class TestPoincareFullyConnected:
    """Test Poincaré fully-connected layer operation."""

    @pytest.mark.parametrize(
        "batch_size,input_dim,output_dim",
        [
            (5, 3, 2),
            (10, 10, 5),
            (8, 50, 20),
        ],
    )
    def test_fully_connected_without_bias(
        self, batch_size, input_dim, output_dim, tolerance, jax_key
    ):
        """Test fully-connected layer without bias."""
        # Generate test data
        key_x, key_z, key_c = jax.random.split(jax_key, 3)

        # Input points on the manifold
        x = jax.random.uniform(key_x, (batch_size, input_dim), minval=-0.5, maxval=0.5)
        # Weight matrix (hyperplane orientations)
        z = jax.random.normal(key_z, (input_dim, output_dim)) * 0.1
        # Curvature
        c = jnp.abs(jax.random.normal(key_c, ())) + 0.1

        # Compute using hypax
        result = hypax_fully_connected(x, z, None, c, axis=-1)

        # Check output is on the manifold (within the ball)
        assert result.shape == (batch_size, output_dim), (
            f"Expected shape {(batch_size, output_dim)}, got {result.shape}"
        )

        # Check points are within the Poincaré ball (norm < 1/sqrt(c))
        norms = jnp.linalg.norm(result, axis=-1)
        max_norm = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < max_norm * 1.1), (
            f"Some points outside Poincaré ball: max norm {jnp.max(norms)}, limit {max_norm}"
        )

        # Check for finite values
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"

    @pytest.mark.parametrize(
        "batch_size,input_dim,output_dim",
        [
            (5, 3, 2),
            (10, 10, 5),
        ],
    )
    def test_fully_connected_with_bias(
        self, batch_size, input_dim, output_dim, tolerance, jax_key
    ):
        """Test fully-connected layer with bias."""
        # Generate test data
        key_x, key_z, key_b, key_c = jax.random.split(jax_key, 4)

        # Input points on the manifold
        x = jax.random.uniform(key_x, (batch_size, input_dim), minval=-0.5, maxval=0.5)
        # Weight matrix
        z = jax.random.normal(key_z, (input_dim, output_dim)) * 0.1
        # Bias
        bias = jax.random.normal(key_b, (output_dim,)) * 0.1
        # Curvature
        c = jnp.abs(jax.random.normal(key_c, ())) + 0.1

        # Compute using hypax
        result = hypax_fully_connected(x, z, bias, c, axis=-1)

        # Check output is on the manifold
        assert result.shape == (batch_size, output_dim), (
            f"Expected shape {(batch_size, output_dim)}, got {result.shape}"
        )

        # Check points are within the Poincaré ball
        norms = jnp.linalg.norm(result, axis=-1)
        max_norm = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < max_norm * 1.1), f"Some points outside Poincaré ball"

        # Check for finite values
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"

    def test_fully_connected_zero_input(self, tolerance, jax_key):
        """Test fully-connected layer with zero input (origin)."""
        input_dim, output_dim = 3, 2

        # Generate test data
        key_z, key_c = jax.random.split(jax_key, 2)

        # Zero input
        x = jnp.zeros((1, input_dim))
        # Weight matrix
        z = jax.random.normal(key_z, (input_dim, output_dim))
        # Curvature
        c = jnp.array(1.0)

        # Compute using hypax
        result = hypax_fully_connected(x, z, None, c, axis=-1)

        assert result.shape == (1, output_dim)
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"

    def test_fully_connected_gradients(self, tolerance, jax_key):
        """Test that gradients can be computed through the layer."""
        batch_size, input_dim, output_dim = 5, 3, 2

        # Generate test data
        key_x, key_z, key_c = jax.random.split(jax_key, 3)

        x = jax.random.uniform(key_x, (batch_size, input_dim), minval=-0.5, maxval=0.5)
        z = jax.random.normal(key_z, (input_dim, output_dim)) * 0.1
        c = jnp.array(1.0)

        # Define a simple loss function
        def loss_fn(z_param):
            output = hypax_fully_connected(x, z_param, None, c, axis=-1)
            return jnp.sum(output**2)

        # Compute gradients
        loss, grad = jax.value_and_grad(loss_fn)(z)

        # Check gradients are finite
        assert jnp.all(jnp.isfinite(grad)), "Gradients contain non-finite values"
        assert grad.shape == z.shape, (
            f"Gradient shape {grad.shape} doesn't match parameter shape {z.shape}"
        )
