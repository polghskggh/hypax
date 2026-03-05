"""Shared fixtures and configuration for hypax tests."""

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture(scope="session")
def jax_key():
    """JAX random key fixture."""
    return jax.random.key(0)


@pytest.fixture
def tolerance():
    """Numerical tolerance for comparisons."""
    return {"rtol": 1e-5, "atol": 1e-6}


@pytest.fixture(
    params=[
        (3,),
        (10,),
        (50,),
    ]
)
def shape(request):
    """Common test shapes for vectors."""
    return request.param


@pytest.fixture(
    params=[
        (1, 3),
        (1, 30),
        (1, 300),
    ]
)
def batch_shape(request):
    """Common batch shapes for testing."""
    return request.param


@pytest.fixture(
    params=[
        [0.1, 0.1, 0.1],
        [0.01, 0.41, 0.12],
        [1.0, 1.0, 1.0],
    ]
)
def curvature(request):
    """Common curvature values for testing."""
    return request.param


@pytest.fixture(
    params=[
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
    ]
)
def test_vector(request):
    """Common test vectors."""
    return request.param
