# Mathematical utility functions for hyperbolic operations

import jax.numpy as jnp
from jax.scipy.special import gammaln
import jax

def beta_func(a: float | jnp.ndarray, b: float | jnp.ndarray) -> jnp.ndarray:
    """Compute the beta function B(a, b).

    The beta function is defined as:
        B(a, b) = Γ(a)Γ(b) / Γ(a+b)

    where Γ is the gamma function. We compute it using the log-gamma function
    for numerical stability:
        B(a, b) = exp(ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b)))

    This is used in beta-concatenation for combining hyperbolic vectors when
    increasing dimensionality (e.g., in convolution unfold operations).

    Args:
        a: First parameter (positive real number)
        b: Second parameter (positive real number)

    Returns:
        The value of B(a, b)

    References:
        Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        https://aclanthology.org/2022.acl-long.389/
    """
    return jnp.exp(gammaln(a) + gammaln(b) - gammaln(a + b))

def _ensure_batch_dim(tensor: jax.Array) -> tuple[jax.Array, bool]:
    if tensor.ndim == 3:
        return tensor, False
    if tensor.ndim == 2:
        return tensor[jnp.newaxis, ...], True
    raise ValueError("Expected tensor with 2 or 3 dimensions")

