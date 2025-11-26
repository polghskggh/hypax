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
from hypax.manifolds.curvature import Curvature

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
        x = hrelu(x, axis=1)
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
        # # Hyperbolic linear layers
        self.linear1 = HLinear(
            64 * 7 * 7, 256, manifold=manifold, rngs=rngs
        )
        self.linear2 = HLinear(256, 10, manifold=manifold, rngs=rngs)

    def __call__(self, x):
        # Input x should be a regular JAX array, wrap it in ManifoldArray
        x = ManifoldArray(data=x, manifold=self.manifold)

        # Hyperbolic conv + activation
        x = self.conv1(x)
        # jax.debug.print('post-conv {x}', x=jnp.any(jnp.isnan(x.data)))

        x = hrelu(x, axis=1)
        # jax.debug.print('post-hrelu {x}', x=jnp.any(jnp.isnan(x.data)))

        x = self.pool(x)
        # jax.debug.print('post-pool {x}', x=jnp.any(jnp.isnan(x.data)))

        x = self.conv2(x)
        # jax.debug.print('post-conv2 {x}', x=jnp.any(jnp.isnan(x.data)))

        x = hrelu(x, axis=1)
        # jax.debug.print('post-hrelu2 {x}', x=jnp.any(jnp.isnan(x.data)))

        x = self.pool(x)
        # jax.debug.print('post-pool2 {x}', x=jnp.any(jnp.isnan(x.data)))


        # Flatten to (batch_size, 64*7*7)
        batch_size = x.shape[0]
        x = x.replace(data=x.data.reshape(batch_size, -1))
        # jax.debug.print('post_flatten: {x}', x=x.data[0][:10])

        # Hyperbolic linear layers
        x = self.linear1(x)
        x = hrelu(x)
        x = self.linear2(x)

        # Return the underlying array for loss computation
        return x.data


print("Creating model...")
# Create the Poincaré ball manifold with curvature c=1.0
manifold = PoincareBall(curvature=Curvature(1.0, learnable=True))

# Instantiate the hyperbolic model
model = HyperbolicCNN(rngs=nnx.Rngs(0), manifold=manifold)
learning_rate = 0.001

optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: HyperbolicCNN, image, label):
    logits = model(image)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label
    ).mean()
    jax.debug.print('logits {logits}, loss: {loss}', logits=logits, loss=loss)
    return loss, logits

@nnx.jit
def train_step(model: HyperbolicCNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    inputs, labels = batch['image'], batch['label']
    inputs = jnp.expand_dims(inputs, 1)
    inputs /= 1.0
    # inputs = (inputs - 0.13) / 0.30
    manifold_inputs = model.manifold.expmap(inputs, axis=1)
    (loss, logits), grads = grad_fn(model, manifold_inputs, labels)
    metrics.update(loss=loss, logits=logits, labels=labels)
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: HyperbolicCNN, metrics: nnx.MultiMetric, batch):
    inputs, labels = batch['image'], batch['label']
    # inputs = (inputs - 0.13) / 0.30
    inputs = jnp.expand_dims(inputs, 1)

    manifold_inputs = model.manifold.expmap(inputs, axis=1)
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

# TORCH:
# BIAS GRADS: tensor([-4.6478e-02, 9.0341e-02, 3.8723e-02, 7.8561e-02, 2.7807e-02,
#                     -4.9353e-02, -4.8446e-02, -4.9340e-02, -4.4863e-02, -9.2744e-18])
# WEIGHT GRADS: tensor([[0.0113, -0.0020, -0.0075, -0.0120, -0.0014, 0.0063, 0.0056, 0.0063, 0.0042, 0.0020],
# [0.0036, -0.0050, -0.0072, -0.0118, -0.0010, 0.0060, 0.0054, 0.0063, 0.0041, 0.0021],
# [0.0037, -0.0019, -0.0257, -0.0123, -0.0017, 0.0058, 0.0055, 0.0063, 0.0040, 0.0021],
# [0.0034, -0.0022, -0.0071, -0.0408, -0.0015, 0.0062, 0.0055, 0.0063, 0.0041, 0.0019],
# [0.0033, -0.0017, -0.0072, -0.0111, -0.0035, 0.0057, 0.0052, 0.0060, 0.0037, 0.0020],
# [0.0031, -0.0024, -0.0076, -0.0117, -0.0019, 0.0192, 0.0051, 0.0057, 0.0033, 0.0019],
# [0.0032, -0.0026, -0.0080, -0.0125, -0.0020, 0.0057, 0.0174, 0.0060, 0.0038, 0.0022],
# [0.0029, -0.0021, -0.0082, -0.0122, -0.0016, 0.0052, 0.0048, 0.0192, 0.0031, 0.0021],
# [0.0028, -0.0028, -0.0086, -0.0124, -0.0021, 0.0053, 0.0049, 0.0053, 0.0088, 0.0020]]),
# LOGITS: [0.0774, 0.0728, 0.0735, 0.0728, 0.0702, 0.0729, 0.0737, 0.0728, 0.0772, 0.0000]], grad_fn=<MeanBackward1>)


#  JAX
# BIAS GRADS: [-4.6479132e-02  9.0341926e-02  3.8725220e-02  7.8562982e-02 2.7807916e-02
#              -4.9354564e-02 -4.8447222e-02 -4.9342021e-02 -4.4864208e-02 -9.2743374e-18]
# WEIGHT GRADS: [[ 0.01126474 -0.00196714 -0.00754806 -0.01196867 -0.00142827  0.00633148
#                  0.00555898  0.00629176  0.00422809         nan]
#  [ 0.00363716 -0.00495086 -0.00723315 -0.0118125  -0.00102077  0.00599822
#    0.00543366  0.00625763  0.00409878         nan]
#  [ 0.00369832 -0.00191914 -0.02567862 -0.01232831 -0.00171635  0.00582672
#    0.00548796  0.0062756   0.00397613         nan]
#  [ 0.00336749 -0.00222825 -0.00707952 -0.0408021  -0.00146796  0.00615509
#    0.0055255   0.00630853  0.00409341         nan]
#  [ 0.00329371 -0.00170682 -0.00719221 -0.01113134 -0.00354243  0.0057369
#    0.00518157  0.00596579  0.00373821         nan]
#  [ 0.00312696 -0.00244351 -0.00760342 -0.01169285 -0.00187108  0.01922658
#    0.00510577  0.00569719  0.00326725         nan]
#  [ 0.00316407 -0.00258698 -0.00804491 -0.01250922 -0.0019694   0.00571145
#    0.01741341  0.00597756  0.00376567         nan]
#  [ 0.00294589 -0.00210554 -0.00821059 -0.01218478 -0.00156649  0.00521433
#    0.00481311  0.01918483  0.00311122         nan]
#  [ 0.0028212  -0.00279109 -0.00855568 -0.01243486 -0.0021144   0.00531532
#    0.00490668  0.00529856  0.00883825         nan]]
# LOGITS [[0.0752478  0.07211582 0.07357839 0.06974331 0.06741784 0.06946205, 0.07326082 0.07201237 0.07487961 0.        ]
