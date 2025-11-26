import hypll
import hypll.manifolds.poincare_ball as pcb
import hypll.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from flax.nnx import Rngs
from hypll.manifolds.euclidean import Euclidean
from hypll.optim import RiemannianAdam
from hypll.tensors import ManifoldParameter
from hypll.tensors import TangentTensor
from jax._src.tree_util import tree_flatten_with_path

from hypax.array import ManifoldArray
from hypax.manifolds.curvature import Curvature
from hypax.manifolds.poincare_ball import PoincareBall
from hypax.nn.convolution import HConvolution2D
from hypax.opt import riemannian_adam
from hypax.nn.linear import HLinear
from hypax.manifolds.poincare_ball._linalg import poincare_fully_connected, poincare_hyperplane_dists

jax.config.update("jax_enable_x64", True)

def test_jax(inputi):
    manifold = PoincareBall(curvature=Curvature(1.0))
    jax_linear = HLinear(in_features=3, out_features=5, manifold=manifold, rngs=Rngs(0))
    # jax_conv = HConvolution2D(in_channels=3, out_channels=5, kernel_size=3, manifold=manifold, rngs=Rngs(0))
    optimizer = nnx.Optimizer(jax_linear, riemannian_adam(1e-3), wrt=nnx.Param)
    w, b = jax_linear.weights.value, jax_linear.bias.value
    def loss_fn(model, input):
        out = poincare_hyperplane_dists(
            x=input.data, z=model.weights.value, r=None, c=input.manifold.curvature()
        )
        # out = model(input).data
        jax.debug.print('out {o}', o=out)
        return jnp.mean(out)

    @nnx.jit
    def fwd(model, optimizer, inputi):
        input = manifold.expmap(inputi, axis=1)
        input = ManifoldArray(input, manifold=manifold)
        grads = jax.grad(loss_fn)(model, input)
        optimizer.update(model, grads)
        return grads

    for _ in range(1):
        jax_linear = fwd(jax_linear, optimizer, inputi)
        # 0s IN OUTPUT BREAK GRAD FLOW

    for path, leaf in tree_flatten_with_path(jax_linear)[0]:
        print("Path:", path, "Leaf:", leaf)

    def print_leaf(path, leaf):
        jax.debug.print('path: {path_str}, grads: {g}', path_str=path, g=jnp.sum(leaf))
        return leaf  # Return the leaf unchanged

    print(f'POST UPDATE {jnp.sum(jax_linear.weights.value)}, {jnp.sum(jax_linear.bias.value)}')

    return w, b

def test_torch(inputi, weight, bias):
    manifold = pcb.PoincareBall(pcb.Curvature(1.0))
    # torch_conv = nn.modules.convolution.HConvolution2d(in_channels=3, out_channels=5, kernel_size=3, manifold=manifold)
    torch_linear = nn.modules.linear.HLinear(in_features=3, out_features=5, manifold=manifold)
    optimizer = RiemannianAdam(torch_linear.parameters(), lr=1e-3)

    for i in range(1):
        input = TangentTensor(inputi, manifold=manifold, man_dim=1)
        input = manifold.expmap(input)
        # out = torch_linear(input).tensor
        out = hypll.manifolds.poincare_ball.math.linalg.poincare_hyperplane_dists(
            x=input.tensor, z=torch_linear.z.tensor, r=None, c=input.manifold.c()
        )
        print("TORCH", out)
        torch.mean(out).backward()
        print('TORCH WEIGHT',torch_linear.z.tensor.grad, 'TORCH BIAS', torch_linear.bias.grad)
        optimizer.step()

    return out

if __name__ == '__main__':
    input = torch.rand(1, 3)
    weight, bias = test_jax(jnp.array(input))
    out2 = test_torch(input, weight, bias)

# JAX
# Bias: [-2.2089598e-01 -2.2056706e-01 -7.1925245e-02 -5.5171375e-16 -5.5171375e-16]
# Weights: [[0.04662199 0.04326872 0.0141096         nan        nan]
#           [0.04359259 0.04686746 0.01419405        nan        nan]
#           [0.16078743 0.16054802 0.08186265        nan        nan]]
# TORCH
# Bias: [-2.2090e-01, -2.2057e-01, -7.1925e-02, -5.5171e-16, -5.5171e-16]
# Weights: None
# OUTPUT: [0.1353, 0.1361, 0.5020, 0.0000, 0.0000]

# GRADs are nan ->

#