import time

import jax
from flax import nnx
import optax
from datasets import load_dataset
import jax.numpy as jnp
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from tqdm.auto import tqdm

from hypax.manifolds._base import Curvature
from hypax.utils.data import NumpyLoader
from hypax.opt import riemannian_adam
from hypax.manifolds.poincare_ball import PoincareBall
from hypax.array import ManifoldArray
from hypax.nn import HAvgPool2D, HConvolution2D, HLinear, hrelu

print("Loading dataset...")

# Load and preprocess the dataset with batching and channel dimension
dataset = load_dataset("ylecun/mnist").with_format("numpy")

# Batch the datasets
batch_size = 32

train_loader = NumpyLoader(dataset["train"], batch_size=batch_size)
eval_loader = NumpyLoader(dataset["test"], batch_size=batch_size)

eval_ds = dataset["test"].iter(batch_size=batch_size)
# from jax import config
# config.update("jax_enable_x64", True)

class HyperbolicMLP(nnx.Module):
    def __init__(self, rngs, manifold):
        self.rngs = rngs
        self.manifold = manifold
        self.linear1 = HLinear(
             784, 256, manifold=manifold, rngs=rngs
        )
        self.linear2 = HLinear(
            256, 10, manifold=manifold, rngs=rngs
        )
    def __call__(self, x):
        x = ManifoldArray(x.reshape(x.shape[0], -1),
                          self.manifold)
        x = self.linear1(x)
        x = hrelu(x)
        return x.data

class HyperbolicCNN(nnx.Module):
    """A hyperbolic CNN model using hyperbolic layers."""

    def __init__(self, *, rngs: nnx.Rngs, manifold: PoincareBall):
        self.manifold = manifold
        # Hyperbolic convolution layers
        self.conv1 = HConvolution2D(
            1, 32, kernel_size=3, padding=1, manifold=manifold, rngs=rngs
        )
        self.conv2 = HConvolution2D(
            32, 64, kernel_size=3, padding=1, manifold=manifold, rngs=rngs
        )
        self.pool = HAvgPool2D(kernel_size=2, stride=2, manifold=manifold)
        # Hyperbolic linear layers
        self.linear1 = HLinear(
            64 * 7 * 7, 256, manifold=manifold, rngs=rngs
        )
        self.linear2 = HLinear(256, 10, manifold=manifold, rngs=rngs)

    def __call__(self, x):
        # Input x should be a regular JAX array, wrap it in ManifoldArray
        x = ManifoldArray(data=x, manifold=self.manifold)

        # Hyperbolic conv + activation
        x = self.conv1(x)
        x = hrelu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = hrelu(x)
        x = self.pool(x)

        # Flatten to (batch_size, 64*7*7)
        batch_size = x.shape[0]
        x = x.replace(data=x.data.reshape(batch_size, -1))

        # Hyperbolic linear layers
        x = self.linear1(x)
        x = hrelu(x)
        x = self.linear2(x)

        # Return the underlying array for loss computation
        return x.data


print("Creating model...")
# Create the Poincaré ball manifold with curvature c=1.0
manifold = PoincareBall(curvature=Curvature(1.0))

# Instantiate the hyperbolic model
model = HyperbolicMLP(rngs=nnx.Rngs(0), manifold=manifold)
learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: HyperbolicCNN, image, label):
    logits = model(jnp.expand_dims(image, 1))
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: HyperbolicCNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    inputs, labels = batch['image'], batch['label']
    inputs = inputs / 1.0
    manifold_inputs = model.manifold.expmap(inputs)
    # torch [0.0015, 0.3412, 0.0000, 0.0000, 0.4197, 0.3400, 0.0000, 0.0000, 0.0000, 0.0000]
    # jax [0.00281601 0.34403626 0.         0.         0.36831811 0.35981409, 0., 0., 0., 0.]
    (loss, logits), grads = grad_fn(model, manifold_inputs, labels)
    metrics.update(loss=loss, logits=logits, labels=labels)
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: HyperbolicCNN, metrics: nnx.MultiMetric, batch):
    inputs, labels = batch['image'], batch['label']
    inputs = inputs / 1.0
    manifold_inputs = manifold.expmap(inputs)
    loss, logits = loss_fn(model, manifold_inputs, labels)
    metrics.update(loss=loss, logits=logits, labels=labels)


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

eval_every = 5  # Evaluate every 100 steps for efficiency


print("Starting training..")


def train_single_epoch():
    for batch in tqdm(
        train_loader,
        desc="Train",
        leave=False,
        total=len(train_loader),
    ):
        # Convert images to correct shape if necessary (handled in preprocessing)
        train_step(model, optimizer, metrics, batch)


def eval_single_epoch():
    for batch in tqdm(
        eval_loader,
        desc="Eval",
        leave=False,
        total=len(eval_loader),
    ):
        # Convert images to correct shape if necessary (handled in preprocessing)
        eval_step(model, metrics, batch)


num_epochs = 10
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    train_single_epoch()

    msg = f"[{epoch + 1}/{num_epochs}]"

    # Training metrics
    train_metrics = metrics.compute()
    for metric, value in train_metrics.items():
        msg = f"{msg} train_{metric}: {value:.4f}"
        metrics_history[f"train_{metric}"].append(value)
    metrics.reset()

    eval_single_epoch()
    # Eval metrics
    train_metrics = metrics.compute()
    for metric, value in train_metrics.items():
        msg = f"{msg} test_{metric}: {value:.4f}"
        metrics_history[f"test_{metric}"].append(value)
    metrics.reset()

    tqdm.write(msg)
#    JAX train_accuracy: 0.9316 train_loss: 0.2498 test_accuracy: 0.9140 test_loss: 0.3935 1min
#    TORCH train_accuracy: 0.9219 train_loss: 0.3002 test_accuracy: 0.9662 test_loss: 0.1358
# OLD 17min for 10 epochs
