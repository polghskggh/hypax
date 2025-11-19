"""Accuracy tests for differential geometry operations.

Compares hypax (JAX) implementations against hypll (PyTorch) reference implementations.
"""

import pytest
import torch
import jax.numpy as jnp

from hypax.manifolds.poincare_ball._diffgeom import (
    mobius_add as hypax_mobius_add,
    project as hypax_project,
    expmap0 as hypax_expmap0,
    logmap0 as hypax_logmap0,
    expmap as hypax_expmap,
    logmap as hypax_logmap,
    gyration as hypax_gyration,
)
from hypll.manifolds.poincare_ball.math.diffgeom import (
    mobius_add as hypll_mobius_add,
    project as hypll_project,
    expmap0 as hypll_expmap0,
    logmap0 as hypll_logmap0,
    expmap as hypll_expmap,
    logmap as hypll_logmap,
    gyration as hypll_gyration,
)

from tests.conftest import assert_arrays_close


class TestMobiusAdd:
    """Test Möbius addition operation."""

    @pytest.mark.parametrize(
        "x,y,c",
        [
            ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.1, 0.1, 0.1]),
            ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.01, 0.41, 0.12]),
            ([0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [1.0, 1.0, 1.0]),
        ],
    )
    def test_mobius_add_matches_hypll(self, x, y, c, tolerance, torch_device):
        """Test that hypax mobius_add matches hypll implementation."""
        # PyTorch computation
        torch_x = torch.tensor(x, dtype=torch.float32, device=torch_device)
        torch_y = torch.tensor(y, dtype=torch.float32, device=torch_device)
        torch_c = torch.tensor(c, dtype=torch.float32, device=torch_device)
        torch_result = hypll_mobius_add(torch_x, torch_y, torch_c)

        # JAX computation
        jax_x = jnp.array(x, dtype=jnp.float32)
        jax_y = jnp.array(y, dtype=jnp.float32)
        jax_c = jnp.array(c, dtype=jnp.float32)
        jax_result = hypax_mobius_add(jax_x, jax_y, jax_c)

        # Compare results
        assert_arrays_close(
            jax_result,
            torch_result,
            msg="Möbius addition output differs between hypax and hypll",
            **tolerance,
        )


class TestProject:
    """Test projection operation onto Poincaré ball."""

    @pytest.mark.parametrize(
        "x,c",
        [
            ([1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),
            ([1.0, 2.0, 3.0], [0.01, 0.41, 0.12]),
            ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ([10.0, 10.0, 10.0], [0.5, 0.5, 0.5]),  # Test with point far outside ball
        ],
    )
    def test_project_matches_hypll(self, x, c, tolerance, torch_device):
        """Test that hypax project matches hypll implementation."""
        # PyTorch computation
        torch_x = torch.tensor(x, dtype=torch.float32, device=torch_device)
        torch_c = torch.tensor(c, dtype=torch.float32, device=torch_device)
        torch_result = hypll_project(torch_x, torch_c)

        # JAX computation
        jax_x = jnp.array(x, dtype=jnp.float32)
        jax_c = jnp.array(c, dtype=jnp.float32)
        jax_result = hypax_project(jax_x, jax_c)

        # Compare results
        assert_arrays_close(
            jax_result,
            torch_result,
            msg="Project output differs between hypax and hypll",
            **tolerance,
        )


class TestExpmap0:
    """Test exponential map from origin."""

    @pytest.mark.parametrize(
        "x,c",
        [
            ([1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),
            ([1.0, 2.0, 3.0], [0.01, 0.41, 0.12]),
            ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0]),
            ([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),  # Test with zero vector
        ],
    )
    def test_expmap0_matches_hypll(self, x, c, tolerance, torch_device):
        """Test that hypax expmap0 matches hypll implementation."""
        # PyTorch computation
        torch_x = torch.tensor(x, dtype=torch.float32, device=torch_device)
        torch_c = torch.tensor(c, dtype=torch.float32, device=torch_device)
        torch_result = hypll_expmap0(torch_x, torch_c)

        # JAX computation
        jax_x = jnp.array(x, dtype=jnp.float32)
        jax_c = jnp.array(c, dtype=jnp.float32)
        jax_result = hypax_expmap0(jax_x, jax_c)

        # Compare results
        assert_arrays_close(
            jax_result,
            torch_result,
            msg="Expmap0 output differs between hypax and hypll",
            **tolerance,
        )


class TestLogmap0:
    """Test logarithmic map to origin."""

    @pytest.mark.parametrize(
        "x,c",
        [
            ([0.1, 0.1, 0.1], [3.3, 3.3, 3.3]),
            ([0.1, 0.1, 0.1], [1.01, 1.41, 1.12]),
            ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ([0.01, 0.01, 0.01], [0.1, 0.1, 0.1]),  # Test with small values
        ],
    )
    def test_logmap0_matches_hypll(self, x, c, tolerance, torch_device):
        """Test that hypax logmap0 matches hypll implementation."""
        # PyTorch computation
        torch_x = torch.tensor(x, dtype=torch.float32, device=torch_device)
        torch_c = torch.tensor(c, dtype=torch.float32, device=torch_device)
        torch_result = hypll_logmap0(torch_x, torch_c)

        # JAX computation
        jax_x = jnp.array(x, dtype=jnp.float32)
        jax_c = jnp.array(c, dtype=jnp.float32)
        jax_result = hypax_logmap0(jax_x, jax_c)

        # Compare results
        assert_arrays_close(
            jax_result,
            torch_result,
            msg="Logmap0 output differs between hypax and hypll",
            **tolerance,
        )


class TestExpmap:
    """Test exponential map from arbitrary point."""

    @pytest.mark.parametrize(
        "x,v,c",
        [
            ([1.0, 2.0, 3.0], [3.3, 3.3, 3.3], [0.1, 0.1, 0.1]),
            ([1.0, 2.0, 3.0], [1.01, 1.41, 1.12], [0.1, 0.1, 0.1]),
            ([0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.1, 0.1]),  # Test from origin
        ],
    )
    def test_expmap_matches_hypll(self, x, v, c, tolerance, torch_device):
        """Test that hypax expmap matches hypll implementation."""
        # PyTorch computation
        torch_x = torch.tensor(x, dtype=torch.float32, device=torch_device)
        torch_v = torch.tensor(v, dtype=torch.float32, device=torch_device)
        torch_c = torch.tensor(c, dtype=torch.float32, device=torch_device)
        torch_result = hypll_expmap(torch_x, torch_v, torch_c)

        # JAX computation
        jax_x = jnp.array(x, dtype=jnp.float32)
        jax_v = jnp.array(v, dtype=jnp.float32)
        jax_c = jnp.array(c, dtype=jnp.float32)
        jax_result = hypax_expmap(jax_x, jax_v, jax_c)

        # Compare results
        assert_arrays_close(
            jax_result,
            torch_result,
            msg="Expmap output differs between hypax and hypll",
            **tolerance,
        )


class TestLogmap:
    """Test logarithmic map to arbitrary point."""

    @pytest.mark.parametrize(
        "x,v,c",
        [
            ([0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]),
            ([0.1, 0.1, 0.1], [0.01, 0.41, 0.12], [0.1, 0.1, 0.1]),
            ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [1.0, 1.0, 1.0]),
            ([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.1]),  # Test to origin
        ],
    )
    def test_logmap_matches_hypll(self, x, v, c, tolerance, torch_device):
        """Test that hypax logmap matches hypll implementation."""
        # PyTorch computation
        torch_x = torch.tensor(x, dtype=torch.float32, device=torch_device)
        torch_v = torch.tensor(v, dtype=torch.float32, device=torch_device)
        torch_c = torch.tensor(c, dtype=torch.float32, device=torch_device)
        torch_result = hypll_logmap(torch_x, torch_v, torch_c)

        # JAX computation
        jax_x = jnp.array(x, dtype=jnp.float32)
        jax_v = jnp.array(v, dtype=jnp.float32)
        jax_c = jnp.array(c, dtype=jnp.float32)
        jax_result = hypax_logmap(jax_x, jax_v, jax_c)

        # Compare results
        assert_arrays_close(
            jax_result,
            torch_result,
            msg="Logmap output differs between hypax and hypll",
            **tolerance,
        )


class TestGyration:
    """Test gyration (hyperbolic rotation) operation."""

    @pytest.mark.parametrize(
        "u,v,w,c",
        [
            ([0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]),
            ([0.1, 0.1, 0.1], [0.01, 0.41, 0.12], [0.3, 0.2, 0.4], [0.1, 0.1, 0.1]),
            ([0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0]),
            (
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.1, 0.1, 0.1],
            ),  # Test with zero
        ],
    )
    def test_gyration_matches_hypll(self, u, v, w, c, tolerance, torch_device):
        """Test that hypax gyration matches hypll implementation."""
        # PyTorch computation
        torch_u = torch.tensor(u, dtype=torch.float32, device=torch_device)
        torch_v = torch.tensor(v, dtype=torch.float32, device=torch_device)
        torch_w = torch.tensor(w, dtype=torch.float32, device=torch_device)
        torch_c = torch.tensor(c, dtype=torch.float32, device=torch_device)
        torch_result = hypll_gyration(torch_u, torch_v, torch_w, torch_c)

        # JAX computation
        jax_u = jnp.array(u, dtype=jnp.float32)
        jax_v = jnp.array(v, dtype=jnp.float32)
        jax_w = jnp.array(w, dtype=jnp.float32)
        jax_c = jnp.array(c, dtype=jnp.float32)
        jax_result = hypax_gyration(jax_u, jax_v, jax_w, jax_c)

        # Compare results
        assert_arrays_close(
            jax_result,
            torch_result,
            msg="Gyration output differs between hypax and hypll",
            **tolerance,
        )
