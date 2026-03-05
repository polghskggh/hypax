"""Unit tests for ManifoldArray class."""

import pytest
import jax.numpy as jnp

from hypax.array._manifold_array import ManifoldArray
from hypax.manifolds.poincare_ball import PoincareBall
from hypax.manifolds.curvature import Curvature


@pytest.fixture
def poincare_manifold():
    return PoincareBall(Curvature(1.0))


class TestManifoldArray:
    def test_initialization(self, poincare_manifold):
        data = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        ma = ManifoldArray(data, poincare_manifold)
        assert isinstance(ma, ManifoldArray)
        assert jnp.array_equal(ma.data, data)
        assert ma.manifold is poincare_manifold

    def test_shape(self, poincare_manifold):
        assert ManifoldArray(jnp.array([0.1, 0.2, 0.3]), poincare_manifold).shape == (3,)
        assert ManifoldArray(jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), poincare_manifold).shape == (3, 2)
        assert ManifoldArray(jnp.ones((2, 3, 4)), poincare_manifold).shape == (2, 3, 4)

    def test_ndim(self, poincare_manifold):
        assert ManifoldArray(jnp.array([0.1, 0.2]), poincare_manifold).ndim == 1
        assert ManifoldArray(jnp.array([[0.1, 0.2]]), poincare_manifold).ndim == 2
        assert ManifoldArray(jnp.ones((2, 3, 4)), poincare_manifold).ndim == 3

    def test_different_curvatures(self):
        data = jnp.array([[0.1, 0.2, 0.3]])
        for c in [0.5, 1.0, 2.0]:
            manifold = PoincareBall(Curvature(c))
            ma = ManifoldArray(data, manifold)
            assert jnp.array_equal(ma.data, data)

    def test_preserves_dtype(self, poincare_manifold):
        data = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        assert ManifoldArray(data, poincare_manifold).data.dtype == jnp.float32

    def test_empty(self, poincare_manifold):
        ma = ManifoldArray(jnp.array([]), poincare_manifold)
        assert ma.shape == (0,)
        assert len(ma.data) == 0

    def test_zero_vector(self, poincare_manifold):
        ma = ManifoldArray(jnp.zeros((1, 3)), poincare_manifold)
        assert jnp.all(ma.data == 0)
