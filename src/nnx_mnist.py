import time

import jax
from flax import nnx
import optax
from datasets import load_dataset
import jax.numpy as jnp
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from tqdm.auto import tqdm

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
manifold = PoincareBall(c=1.0)

# Instantiate the hyperbolic model
model = HyperbolicCNN(rngs=nnx.Rngs(0), manifold=manifold)
learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: HyperbolicCNN, batch):
    logits = model(jnp.expand_dims(batch["image"], 1))
    jax.debug.print('logit: {l}', l=logits)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: HyperbolicCNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: HyperbolicCNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


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
        start_time = time.time()
        train_step(model, optimizer, metrics, batch)
        print('batch_time: ', time.time() - start_time)


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
# OLD 17min for 10 epochs
