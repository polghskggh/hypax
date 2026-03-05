from typing import Tuple, Sequence

import jax
import jax.numpy as jnp

from hypax.manifolds.poincare_ball._math import expmap0, logmap0
from hypax.utils.math import beta_func


def _pair(value: int | Tuple[int, int] | Sequence[int]) -> Tuple[int, int]:
    """Normalize kernel/stride/padding arguments to 2-tuples."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected tuple of length 2, got {value}")
        return tuple(int(v) for v in value)
    if isinstance(value, Sequence):
        value = list(value)
        if len(value) != 2:
            raise ValueError(f"Expected sequence of length 2, got {value}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))




def unfold_2d(
    x: jax.Array,
    kernel_size: Tuple[int, int],
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
) -> jax.Array:
    """Extract sliding local blocks from a batched 2D input (im2col operation).

    This is the JAX equivalent of PyTorch's torch.nn.functional.unfold.

    Args:
        x: Input array with shape [batch, channels, height, width]
        kernel_size: Size of the sliding blocks as (kernel_h, kernel_w)
        stride: Stride of the sliding blocks (default: 1)
        padding: Implicit zero padding on both sides (default: 0)

    Returns:
        Array with shape [batch, channels * kernel_h * kernel_w, num_patches]
        where num_patches = out_h * out_w
    """
    batch_size, channels, height, width = x.shape
    kernel_h, kernel_w = kernel_size

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        x = jnp.pad(
            x,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
        height += 2 * pad_h
        width += 2 * pad_w

    # Calculate output dimensions
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1

    # Extract patches using strided slicing
    # Create indices for all patches
    patches = []
    # TODO reimplement with jax.lax.scan/batchify
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            w_start = j * stride_w
            patch = x[:, :, h_start : h_start + kernel_h, w_start : w_start + kernel_w]
            # Reshape to [batch, channels * kernel_h * kernel_w]
            patch = patch.reshape(batch_size, -1)
            patches.append(patch)

    # Stack all patches: [batch, channels * kernel_h * kernel_w, num_patches]
    output = jnp.stack(patches, axis=-1)

    return output
