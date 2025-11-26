import jax
import jax.numpy as jnp
import torch

def jax_with_zeros(x):
    def loss_fn(weights, x):
        # Example loss function
        return jnp.sum((x @ weights) ** 2)
    weights = jnp.zeros((10,5))  # Initialize weights to zero
    grad_fn = jax.grad(loss_fn)

    # Compute gradients
    grads = grad_fn(weights, x)
    print(grads)


def torch_with_zeros(x):
    weights = torch.zeros((10,5), requires_grad=True)
    loss = torch.sum((x @ weights) ** 2)
    loss.backward()
    print(weights.grad)


if __name__ == "__main__":
    x = torch.randn(10)
    jax_with_zeros(jnp.asarray(x.detach().numpy()))
    torch_with_zeros(x)