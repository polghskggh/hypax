 """Unit tests for ManifoldArray and TangentArray classes."""

import pytest
import jax.numpy as jnp

from hypax.array._manifold_array import ManifoldArray
from hypax.array._tangent_array import TangentArray
from hypax.manifolds.poincare_ball import PoincareBall


@pytest.fixture
def poincare_manifold():
    """Create a PoincareBall manifold for testing."""
    return PoincareBall(c=1.0)


class TestManifoldArray:
    """Test ManifoldArray wrapper class."""

    def test_manifold_array_initialization(self, poincare_manifold, jax_key):
        """Test basic initialization of ManifoldArray."""
        data = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        man_array = ManifoldArray(data, poincare_manifold)

        assert isinstance(man_array, ManifoldArray)
        assert jnp.array_equal(man_array.array, data)
        assert man_array.manifold == poincare_manifold

    def test_manifold_array_shape(self, poincare_manifold):
        """Test shape property of ManifoldArray."""
        # Test 1D
        data_1d = jnp.array([0.1, 0.2, 0.3])
        man_array_1d = ManifoldArray(data_1d, poincare_manifold)
        assert man_array_1d.shape == (3,)

        # Test 2D
        data_2d = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        man_array_2d = ManifoldArray(data_2d, poincare_manifold)
        assert man_array_2d.shape == (3, 2)

        # Test 3D
        data_3d = jnp.ones((2, 3, 4))
        man_array_3d = ManifoldArray(data_3d, poincare_manifold)
        assert man_array_3d.shape == (2, 3, 4)

    def test_manifold_array_ndim(self, poincare_manifold):
        """Test ndim property of ManifoldArray."""
        # Test different dimensions
        data_1d = jnp.array([0.1, 0.2, 0.3])
        man_array_1d = ManifoldArray(data_1d, poincare_manifold)
        assert man_array_1d.ndim == 1

        data_2d = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        man_array_2d = ManifoldArray(data_2d, poincare_manifold)
        assert man_array_2d.ndim == 2

        data_3d = jnp.ones((2, 3, 4))
        man_array_3d = ManifoldArray(data_3d, poincare_manifold)
        assert man_array_3d.ndim == 3

    def test_manifold_array_with_different_manifolds(self, jax_key):
        """Test ManifoldArray with different manifold curvatures."""
        data = jnp.array([[0.1, 0.2, 0.3]])

        for c_value in [0.5, 1.0, 2.0]:
            manifold = PoincareBall(c=c_value)
            man_array = ManifoldArray(data, manifold)

            assert man_array.manifold.curvature.value == c_value
            assert jnp.array_equal(man_array.array, data)

    def test_manifold_array_preserves_dtype(self, poincare_manifold):
        """Test that ManifoldArray preserves float32 dtype."""
        # Test float32 (JAX default dtype)
        data_f32 = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        man_array_f32 = ManifoldArray(data_f32, poincare_manifold)
        assert man_array_f32.array.dtype == jnp.float32

    def test_manifold_array_empty(self, poincare_manifold):
        """Test ManifoldArray with empty array."""
        data = jnp.array([])
        man_array = ManifoldArray(data, poincare_manifold)

        assert man_array.shape == (0,)
        assert man_array.ndim == 1
        assert len(man_array.array) == 0

    def test_manifold_array_zero_vector(self, poincare_manifold):
        """Test ManifoldArray with zero vector (origin)."""
        data = jnp.zeros((1, 3))
        man_array = ManifoldArray(data, poincare_manifold)

        assert jnp.all(man_array.array == 0)
        assert man_array.shape == (1, 3)


class TestTangentArray:
    """Test TangentArray wrapper class."""

    def test_tangent_array_initialization(self, jax_key):
        """Test basic initialization of TangentArray."""
        data = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        tan_array = TangentArray(data)

        assert isinstance(tan_array, TangentArray)
        assert jnp.array_equal(tan_array.array, data)

    def test_tangent_array_different_shapes(self):
        """Test TangentArray with different shapes."""
        # Test 1D
        data_1d = jnp.array([0.1, 0.2, 0.3])
        tan_array_1d = TangentArray(data_1d)
        assert tan_array_1d.array.shape == (3,)

        # Test 2D
        data_2d = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        tan_array_2d = TangentArray(data_2d)
        assert tan_array_2d.array.shape == (2, 2)

        # Test 3D
        data_3d = jnp.ones((2, 3, 4))
        tan_array_3d = TangentArray(data_3d)
        assert tan_array_3d.array.shape == (2, 3, 4)

    def test_tangent_array_preserves_dtype(self):
        """Test that TangentArray preserves float32 dtype."""
        # Test float32 (JAX default dtype)
        data_f32 = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        tan_array_f32 = TangentArray(data_f32)
        assert tan_array_f32.array.dtype == jnp.float32

    def test_tangent_array_empty(self):
        """Test TangentArray with empty array."""
        data = jnp.array([])
        tan_array = TangentArray(data)

        assert tan_array.array.shape == (0,)
        assert len(tan_array.array) == 0

    def test_tangent_array_zero_vector(self):
        """Test TangentArray with zero tangent vector."""
        data = jnp.zeros((1, 3))
        tan_array = TangentArray(data)

        assert jnp.all(tan_array.array == 0)
        assert tan_array.array.shape == (1, 3)

    def test_tangent_array_large_values(self):
        """Test TangentArray with large tangent vectors."""
        data = jnp.array([[100.0, 200.0, 300.0]])
        tan_array = TangentArray(data)

        assert jnp.array_equal(tan_array.array, data)
        # Tangent vectors can have arbitrary magnitude
        assert jnp.all(jnp.isfinite(tan_array.array))

    def test_tangent_array_negative_values(self):
        """Test TangentArray with negative values."""
        data = jnp.array([[-1.0, -2.0, -3.0]])
        tan_array = TangentArray(data)

        assert jnp.array_equal(tan_array.array, data)
        assert jnp.all(tan_array.array < 0)


class TestArrayInteraction:
    """Test interactions between ManifoldArray and TangentArray."""

    def test_manifold_and_tangent_array_shape_compatibility(self, poincare_manifold):
        """Test that ManifoldArray and TangentArray can have matching shapes."""
        shape = (5, 3)

        # Create arrays with same shape
        manifold_data = jnp.ones(shape) * 0.1
        tangent_data = jnp.ones(shape) * 0.5

        man_array = ManifoldArray(manifold_data, poincare_manifold)
        tan_array = TangentArray(tangent_data)

        assert man_array.shape == tan_array.array.shape
        assert man_array.ndim == tan_array.array.ndim

    def test_wrapping_jax_operations(self, poincare_manifold):
        """Test that JAX operations can be applied to wrapped arrays."""
        data = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        man_array = ManifoldArray(data, poincare_manifold)
        tan_array = TangentArray(data)

        # Test that we can still perform JAX operations on the underlying arrays
        man_sum = jnp.sum(man_array.array)
        tan_sum = jnp.sum(tan_array.array)

        assert jnp.isfinite(man_sum)
        assert jnp.isfinite(tan_sum)
        assert man_sum == tan_sum  # Same input data
