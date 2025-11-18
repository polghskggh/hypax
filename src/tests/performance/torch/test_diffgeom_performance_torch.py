"""Performance benchmarks for differential geometry operations using PyTorch.

Benchmarks hypll (PyTorch) implementations.
"""

import pytest
import torch

from hypll.manifolds.poincare_ball.math.diffgeom import expmap0 as hypll_expmap0


@pytest.fixture(scope="module")
def gpu_device():
    """Detect GPU device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
    def test_expmap0_torch_performance(
        self, benchmark, shape, num_iters, gpu_device
    ):
        """Benchmark PyTorch (hypll) expmap0 implementation."""
        # Prepare data
        x = torch.rand((num_iters, *shape), device=gpu_device)
        c = torch.rand((num_iters, *shape), device=gpu_device)

        # Warmup
        for _ in range(5):
            _ = hypll_expmap0(x, c)

        # Benchmark
        def run():
            result = hypll_expmap0(x, c)
            if gpu_device in ["cuda", "mps"]:
                torch.cuda.synchronize() if gpu_device == "cuda" else torch.mps.synchronize()
            return result

        benchmark(run)


# Note: To run benchmarks, use:
#   pytest tests/performance/test_diffgeom_performance_pytorch.py --benchmark-only
# To skip benchmarks in regular test runs, use:
#   pytest tests/ -m "not benchmark"
