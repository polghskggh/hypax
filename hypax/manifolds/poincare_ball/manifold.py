# manifold.py
#
# JAX implementation of the Poincaré ball model of hyperbolic space

from __future__ import annotations

import functools
import typing as tp

import jax
import jax.numpy as jnp
from jax._src.lax.convolution import conv_dimension_numbers

from hypax.manifolds._base import Manifold
from hypax.manifolds.curvature import Curvature
from hypax.manifolds.poincare_ball._math import (
    expmap0,
    expmap,
    logmap0,
    logmap,
    project,
    mobius_add,
    safe_norm, mobius_add_batch, gyration,
)
from hypax.manifolds.poincare_ball._math import rescale_norm, poincare_hyperplane_dists
from hypax.manifolds.poincare_ball._stats import (
    _align_axes_for_reduction, _frechet_ball_forward, _restore_axes,
)
from hypax.utils.math import beta_func, _ensure_batch_dim
from jax import lax


class PoincareBall(Manifold):
    """Poincaré ball model of hyperbolic space.

    The Poincaré ball is a conformal model of hyperbolic geometry. Points on
    the manifold are represented as vectors x with ||x|| < 1/√c, where c is
    the curvature.

    Attributes:
        curvature: Curvature of the manifold (scalar > 0). Higher curvature means
                   more "curved" space with a smaller radius.

    Example:
        >>> manifold = PoincareBall(c=1.0)
        >>> # Points must satisfy ||x|| < 1/√c
        >>> x = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        >>> # Use manifold operations like expmap, logmap, etc.
    """

    def __init__(self, curvature: Curvature):
        super().__init__()
        self.curvature = curvature

    def expmap(
        self, v: jax.Array, x: jax.Array | None = None, axis: int = -1
    ) -> jax.Array:
        if x is None:
            return expmap0(v, self.curvature(), axis=axis)
        else:
            return expmap(x, v, self.curvature(), axis=axis)

    def logmap(
        self, y: jax.Array, x: jax.Array | None = None, axis: int = -1
    ) -> jax.Array:
        """Logarithmic map: map from manifold to tangent space.

        Args:
            y: Point on manifold
            x: Base point on manifold (if None, uses origin)
            axis: Manifold dimension axis

        Returns:
            Tangent vector at base point
        """
        if x is None:
            return logmap0(y, self.curvature(), axis=axis)
        else:
            return logmap(x, y, self.curvature(), axis=axis)

    def lambda_(self, squared_norm):
        return  2 / jnp.clip(1 - self.curvature() * squared_norm, min=1e-15)

    def project(self, x: jax.Array, axis: int = -1, eps: float = -1.0) -> jax.Array:
        """Project point to be within the Poincaré ball.

        Args:
            x: Point (possibly outside ball)
            axis: Manifold dimension axis
            eps: Epsilon for numerical stability (if < 0, uses default)

        Returns:
            Point projected to be within ball
        """
        return project(x, self.curvature(), axis=axis, eps=eps)

    def mobius_add(self, x: jax.Array, y: jax.Array, axis: int = -1) -> jax.Array:
        """Möbius addition in the Poincaré ball.

        Args:
            x: First point on manifold
            y: Second point on manifold
            axis: Manifold dimension axis

        Returns:
            Result of Möbius addition
        """
        return mobius_add(x, y, self.curvature(), axis=axis)

    def dist(
        self, x: jax.Array, y: jax.Array, axis: int = -1, keepdims: bool = False
    ) -> jax.Array:
        """Hyperbolic distance between points.

        Args:
            x: First point on manifold
            y: Second point on manifold
            axis: Manifold dimension axis
            keepdims: Whether to keep dimensions

        Returns:
            Hyperbolic distance
        """

        c = self.curvature()
        return (2 / jnp.sqrt(c) * jnp.atanh((jnp.sqrt(c) * safe_norm(mobius_add(-x, y, c, axis=axis), axis=axis,
                                                                     keepdims=keepdims))))

    def cdist(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Pairwise hyperbolic distances between batches of points."""
        c = self.curvature()
        x, squeezed_x = _ensure_batch_dim(x)
        y, squeezed_y = _ensure_batch_dim(y)

        if x.shape[0] != y.shape[0]:
            if x.shape[0] == 1:
                x = jnp.broadcast_to(x, (y.shape[0],) + x.shape[1:])
                squeezed_x = False
            elif y.shape[0] == 1:
                y = jnp.broadcast_to(y, (x.shape[0],) + y.shape[1:])
                squeezed_y = False
            else:
                raise ValueError(
                    f"Cannot broadcast batch dimensions: x has {x.shape[0]}, y has {y.shape[0]}"
                )

        diffs = mobius_add_batch(-x, y, c)
        norm = safe_norm(diffs, axis=-1)
        distances = 2 / jnp.sqrt(c) * jnp.atanh(jnp.sqrt(c) * norm)

        if x.shape[0] == 1 and squeezed_x and squeezed_y:
            return distances[0]
        return distances

    def frechet_mean(
        self,
        x: jax.Array,
        *,
        reduce_axis: int,
        axis: int = -1,
        keepdims: bool = False,
        max_iter: int = 100,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> jax.Array:
        """Compute the Fréchet mean of points along ``reduce_axis``."""
        c = self.curvature()
        permuted, axes, red = _align_axes_for_reduction(x, axis, reduce_axis)
        perm_shape = permuted.shape
        other_shape = perm_shape[:-2]
        num_points = perm_shape[-2]
        dim = perm_shape[-1]

        flat = permuted.reshape((-1, num_points, dim))
        mean_flat = _frechet_ball_forward(flat, c, max_iter=max_iter, rtol=rtol, atol=atol)
        mean = mean_flat.reshape(other_shape + (dim,))
        return _restore_axes(mean, axes, red, keepdims)

    def midpoint(
        self,
        x: jax.Array,
        *,
        reduce_axis: int,
        axis: int = -1,
        keepdims: bool = False,
    ) -> jax.Array:
        """Compute the hyperbolic midpoint along ``reduce_axis``."""
        c = self.curvature()
        permuted, axes, red = _align_axes_for_reduction(x, axis, reduce_axis)
        lam = 2 / jnp.maximum(1 - c * jnp.sum(jnp.square(permuted), axis=-1, keepdims=True), 1e-15)
        numerator = jnp.sum(lam * permuted, axis=-2, keepdims=True)
        denominator = jnp.maximum(jnp.sum(lam - 1, axis=-2, keepdims=True), 1e-15)
        frac = numerator / denominator
        norm = jnp.sum(jnp.square(frac), axis=-1, keepdims=True)
        mid = frac / (1 + jnp.sqrt(jnp.maximum(1 - c * norm, 1e-15)))
        mid = jnp.squeeze(mid, axis=-2)
        return _restore_axes(mid, axes, red, keepdims)

    def inner(
        self,
        x: jax.Array,
        u: jax.Array,
        v: jax.Array,
        axis: int = -1,
        keepdims: bool = False,
    ) -> jax.Array:
        """Riemannian inner product at point ``x`` between tangent vectors ``u`` and ``v``."""
        c = self.curvature()
        broadcast_dim = max(x.ndim, u.ndim, v.ndim)
        axis = axis if axis >= 0 else broadcast_dim + axis
        x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
        lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)
        dot_prod = (u * v).sum(axis=axis, keepdims=keepdims)
        return jnp.square(lambda_x) * dot_prod

    def euc_to_tangent(
        self, x: jax.Array, u: jax.Array, axis: int = -1
    ) -> jax.Array:
        """Project Euclidean tensor ``u`` onto the tangent space at ``x``."""
        broadcast_dim = max(x.ndim, u.ndim)
        axis = axis if axis >= 0 else broadcast_dim + axis
        x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
        lambda_x = 2 / jnp.clip(1 - self.curvature() * x_norm_sq, min=1e-15)
        return u / jnp.square(lambda_x)

    def transp(
        self, x: jax.Array, y: jax.Array, v: jax.Array, axis: int = -1
    ) -> jax.Array:
        """Parallel transport ``v`` from tangent space at ``x`` to tangent space at ``y``."""
        c = self.curvature()
        broadcast_dim = max(x.ndim, y.ndim, v.ndim)
        axis = axis if axis >= 0 else broadcast_dim + axis

        x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
        lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)

        y_norm_sq = jnp.square(y).sum(axis=axis - broadcast_dim + y.ndim, keepdims=True)
        lambda_y = 2 / jnp.clip(1 - c * y_norm_sq, min=1e-15)

        return gyration(y, -x, v, c, axis=axis) * lambda_x / lambda_y

    def construct_dl_parameters(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        key_z: jax.Array | None = None,
        key_bias: jax.Array | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> tp.Tuple[jax.Array, jax.Array | None]:
        """Construct and initialize parameters for deep learning layers.

        This method initializes weights and biases for hyperbolic neural network
        layers following the initialization strategy from HNN++:
        - For square/tall matrices (in_features <= out_features): identity init
        - For wide matrices (in_features > out_features): normal init

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to create bias parameters
            key_z: JAX random key for weight initialization
            key_bias: JAX random key for bias initialization
            dtype: Data type for parameters

        Returns:
            Tuple of (weights, bias) where:
            - weights: Array of shape (in_features, out_features)
            - bias: Array of shape (out_features,) if bias=True, else None

        References:
            Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        """
        if key_z is None:
            key_z = jax.random.PRNGKey(0)

        # Initialize weights using identity or normal initialization
        if in_features <= out_features:
            # Identity initialization for square or tall matrices
            weights = 0.5 * jnp.eye(in_features, out_features, dtype=dtype)
        else:
            # HNN++ initialization for wide matrices
            std = (2 * in_features * out_features) ** -0.5
            weights = (
                jax.random.normal(key_z, shape=(in_features, out_features), dtype=dtype)
                * std
            )

        # Initialize bias to zeros if needed
        if bias:
            bias_value = jnp.zeros(out_features, dtype=dtype)
        else:
            bias_value = None

        return weights, bias_value

    def fully_connected(
        self,
        x: jax.Array,
        z: jax.Array,
        bias: jax.Array | None,
        axis: int = -1,
    ) -> jax.Array:
        """Hyperbolic fully connected layer operation.

        Applies the fully connected transformation in hyperbolic space using
        hyperplane distances and projections.

        Args:
            x: Input array on the manifold
            z: Weight matrix (tangent space)
            bias: Bias vector (if None, no bias)
            axis: Manifold dimension axis

        Returns:
            Output array on the manifold

        References:
            Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        """
        c = self.curvature()
        c_sqrt = jnp.sqrt(c)
        x = poincare_hyperplane_dists(x=x, z=z, r=bias, c=c, axis=axis)
        x = jnp.sinh(c_sqrt * x) / c_sqrt

        result = x / (1 + jnp.sqrt(1 + c * jnp.pow(x, 2).sum(axis=axis, keepdims=True)))
        return self.project(result, axis=axis)

    def flatten(self, x, manifold_axis, start_axis=1, end_axis=-1):
        manifold_axis = x.ndim + manifold_axis if manifold_axis < 0 else manifold_axis
        start_axis = x.ndim + start_axis if start_axis < 0 else start_axis
        end_axis = x.ndim + end_axis if end_axis < 0 else end_axis

        flattened_shape = functools.reduce(lambda a, b: a * b, x.shape[start_axis: end_axis+1])
        new_shape = x.shape[:start_axis] + (flattened_shape, ) + x.shape[end_axis+1:]
        if start_axis <= manifold_axis <= end_axis:
            # Use beta concatenation to flatten the manifold dimension of the tensor.
            # Start by applying logmap at the origin and computing the betas.
            x = self.logmap(x, axis=manifold_axis)
            n_i = x.shape[manifold_axis]
            n = flattened_shape
            beta_n = beta_func(n / 2, 0.5)
            beta_n_i = beta_func(n_i / 2, 0.5)
            # Flatten the tensor and rescale.
            x = jnp.reshape(x, new_shape)
            x *= beta_n / beta_n_i
            new_manifold_axis = start_axis
            # Apply exponential map at the origin.
            x = self.expmap(x, axis=new_manifold_axis)
            return x
        else:
            flattened = jnp.reshape(x, new_shape)
            return flattened

    def unfold(self, x: jax.Array, kernel_size: tuple, channels, stride, padding, axis=0):
        kernel_h, kernel_w = kernel_size
        kernel_vol = kernel_h * kernel_w

        x_tangent = self.logmap(x, axis=axis)
        x_tangent = rescale_norm(x_tangent, channels, channels * kernel_vol)
        format = ("NHWC", "HWIO", "NHWC")
        dnums = conv_dimension_numbers(x.shape, (kernel_h, kernel_w, x.shape[-1], channels), format)
        patches = lax.conv_general_dilated_patches(x_tangent, kernel_size, window_strides=stride, padding=padding,
                                                   dimension_numbers=dnums)
        x_manifold = self.expmap(patches, axis=axis)
        return x_manifold
