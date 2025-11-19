"""Shared fixtures and configuration for hypax tests."""

import pytest
import torch
import jax
import jax.numpy as jnp


# Device detection for PyTorch
def get_torch_device():
    """Detect available PyTorch device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture(scope="session")
def torch_device():
    """PyTorch device fixture."""
    return get_torch_device()


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


def jax_to_torch(arr, device="cpu", dtype=torch.float32):
    """Convert JAX array to PyTorch tensor."""
    return torch.tensor(arr, device=device, dtype=dtype)


def torch_to_jax(tensor, dtype=jnp.float32):
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(tensor.cpu().numpy(), dtype=dtype)


def assert_arrays_close(jax_arr, torch_tensor, rtol=1e-5, atol=1e-6, msg=""):
    """Assert JAX array and PyTorch tensor are numerically close."""
    torch_numpy = (
        torch_tensor.cpu().numpy()
        if torch_tensor.is_cuda or torch_tensor.is_mps
        else torch_tensor.numpy()
    )
    assert jnp.allclose(jax_arr, torch_numpy, rtol=rtol, atol=atol, equal_nan=True), (
        f"{msg}\nExpected (torch): {torch_numpy}\nGot (jax): {jax_arr}\n"
        f"Max diff: {jnp.max(jnp.abs(jax_arr - torch_numpy))}"
    )


@pytest.fixture
def array_comparison_helper(tolerance):
    """Helper for comparing arrays with default tolerance."""

    def compare(jax_arr, torch_tensor, msg=""):
        assert_arrays_close(jax_arr, torch_tensor, msg=msg, **tolerance)

    return compare
