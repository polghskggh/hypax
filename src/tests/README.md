# Hypax Test Suite

This directory contains the comprehensive test suite for `hypax`

## Directory Structure

```text
tests/
├── conftest.py                     # Shared fixtures and utilities
├── accuracy/                       # Accuracy tests (hypax vs hypll)
│   ├── test_diffgeom_accuracy.py   # Differential geometry operations
│   ├── test_linalg_accuracy.py     # Linear algebra operations
│   └── test_nn_accuracy.py         # Neural network layers
├── performance/                    # Performance benchmarks
│   └── test_diffgeom_performance.py
└── unit/                           # Unit tests
    └── test_array_classes.py       # ManifoldArray and TangentArray tests
```

## Setup

First, ensure pytest and dependencies are installed:

```bash
# Install dev dependencies including pytest
uv sync --group dev
```

## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run specific test categories

**Accuracy tests only:**

```bash
pytest tests/accuracy/
```

**Performance benchmarks only:**

```bash
pytest tests/performance/ --benchmark-only
```

**Unit tests only:**

```bash
pytest tests/unit/
```

## Running benchmarks

Benchmarks comparing execution speed between JAX and PyTorch implementations.

- **test_diffgeom_performance.py**: Benchmarks expmap0 operation
  - Different input shapes
  - Batch processing comparisons
  - GPU vs CPU performance

**Running benchmarks:**

```bash
# Run all benchmarks
pytest tests/performance/ --benchmark-only

# Run benchmarks with comparison
pytest tests/performance/ --benchmark-compare

# Save benchmark results
pytest tests/performance/ --benchmark-save=my_benchmark
```
