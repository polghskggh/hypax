"""Performance benchmarks for differential geometry operations using JAX.

Benchmarks hypax (JAX) implementations.
"""

import pytest
import jax
import jax.numpy as jnp

from hypax.manifolds.poincare_ball._diffgeom import expmap0 as hypax_expmap0


@pytest.fixture(scope="module")
def jax_key():
    """JAX random key for benchmarks."""
    return jax.random.key(0)


class TestExpmap0Performance:
    """Benchmark expmap0 operation."""

    @pytest.mark.benchmark(group="expmap0")
    @pytest.mark.parametrize(
        "shape,num_iters",
        [
            ((1, 3), 1_000),
            ((1, 30), 1_000),
            ((1, 300), 1_000),
            ((1, 3000), 1_000),
        ],
    )
    def test_expmap0_jax_performance(self, benchmark, shape, num_iters, jax_key):
        """Benchmark JAX (hypax) expmap0 implementation."""

        # Create a wrapper that fixes the axis parameter
        def expmap0_fixed_axis(v, c):
            return hypax_expmap0(v, c, axis=-1)

        # Create vmapped version for batch processing
        expmap_fn = jax.vmap(expmap0_fixed_axis, in_axes=(0, 0))
        expmap_fn_jit = jax.jit(expmap_fn)

        # Prepare data
        key_x, key_c = jax.random.split(jax_key, 2)
        x = jax.random.uniform(key_x, (num_iters, *shape))
        c = jax.random.uniform(key_c, (num_iters, *shape))

        # Warmup (JIT compilation)
        for _ in range(5):
            _ = jax.block_until_ready(expmap_fn_jit(x, c))

        # Benchmark
        def run():
            result = expmap_fn_jit(x, c)
            return jax.block_until_ready(result)

        benchmark(run)


# Note: To run benchmarks, use:
#   pytest tests/performance/test_diffgeom_performance_jax.py --benchmark-only
# To skip benchmarks in regular test runs, use:
#   pytest tests/ -m "not benchmark"
