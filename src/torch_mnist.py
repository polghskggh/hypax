import json
import time

import jax
import torch
from flax import nnx
import optax
from datasets import load_dataset
import jax.numpy as jnp
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from hypll.optim import RiemannianAdam
from hypll.manifolds.poincare_ball import PoincareBall
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.nn.modules.convolution import HConvolution2d
from hypll.nn.modules.linear import HLinear
from hypll.nn.modules.activation import HReLU

from hypll.manifolds.poincare_ball import Curvature

from hypll.nn import HAvgPool2d, HMaxPool2d

print("Loading dataset...")

# Load and preprocess the dataset with batching and channel dimension
dataset = load_dataset("ylecun/mnist").with_format("numpy")

# Batch the datasets
batch_size = 32

train_loader = DataLoader(dataset["train"], batch_size=batch_size)
eval_loader = DataLoader(dataset["test"], batch_size=batch_size)

eval_ds = dataset["test"].iter(batch_size=batch_size)
# from jax import config
# config.update("jax_enable_x64", True)


class HyperbolicCNN(torch.nn.Module):
    """A hyperbolic CNN model using hyperbolic layers."""

    def __init__(self, *args, manifold: PoincareBall, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        # Hyperbolic convolution layers
        self.conv1 = HConvolution2d(
            1, 32, kernel_size=3, padding=1, manifold=manifold
        )
        self.conv2 = HConvolution2d(
            32, 64, kernel_size=3, padding=1, manifold=manifold
        )
        self.pool = HMaxPool2d(kernel_size=2, stride=2, manifold=manifold)
        # Hyperbolic linear layers
        self.linear1 = HLinear(
            64 * 7 * 7, 256, manifold=manifold
        )
        self.linear2 = HLinear(256, 10, manifold=manifold)
        self.activation_fn = HReLU(self.manifold)

    def __call__(self, x):
        # Hyperbolic conv + activation
        x = self.conv1(x)
        x = self.activation_fn(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.pool(x)
        print(x.tensor[0, :, 1, 1])
        # Flatten to (batch_size, 64*7*7)
        x = x.flatten(start_dim=1)
        print(x.tensor[0, :40])
        # Hyperbolic linear layers
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)

        # Return the underlying array for loss computation
        return x.tensor


print("Creating model...")
# Create the Poincaré ball manifold with curvature c=1.0
manifold = PoincareBall(Curvature(1.0))

# Instantiate the hyperbolic model
model = HyperbolicCNN(manifold=manifold)
with open('weights.json', 'r', encoding='utf-8') as f:
    params = json.load(f)
    model.conv1.weights.tensor = torch.nn.Parameter(torch.tensor(params['conv1_w']))
    model.conv1.bias = torch.nn.Parameter(torch.tensor(params['conv1_b']))
    model.conv2.weights.tensor = torch.nn.Parameter(torch.tensor(params['conv2_w']))
    model.conv2.bias = torch.nn.Parameter(torch.tensor(params['conv2_b']))
    model.linear1.z.tensor = torch.nn.Parameter(torch.tensor(params['linear1_w']))
    model.linear1.bias = torch.nn.Parameter(torch.tensor(params['linear1_b']))

    model.linear2.z.tensor = torch.nn.Parameter(torch.tensor(params['linear2_w']))
    model.linear2.bias = torch.nn.Parameter(torch.tensor(params['linear2_b']))


learning_rate = 0.001

optim = RiemannianAdam(model.parameters(), learning_rate)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)
criterion = nn.CrossEntropyLoss()
def loss_fn(model: HyperbolicCNN, image, label):
    logits = model(image)
    loss = criterion(logits, label)
    return loss, logits

def preprocess(batch):
    inputs, labels = batch['image'], batch['label']
    inputs = (inputs -0.13 ) / 0.3
    inputs = inputs[:, None, :, :]
    inputs = TangentTensor(inputs, manifold=manifold, man_dim=1)
    manifold_inputs = manifold.expmap(inputs)
    return manifold_inputs, labels

def train_step(model, optimizer, metrics: nnx.MultiMetric, batch):
    manifold_inputs, labels = preprocess(batch)

    optimizer.zero_grad()
    loss, logits = loss_fn(model, manifold_inputs, labels)
    loss.backward()
    print(loss)
    # print("TORCH LOGITS", logits)
    # for name, p in model.named_parameters():
    #     print(name, p.grad)
    optimizer.step()
    exit(0)
    metrics.update(loss=loss.detach().numpy(), logits=logits.detach().numpy(), labels=labels.detach().numpy())


def eval_step(model, metrics: nnx.MultiMetric, batch):
    manifold_inputs, labels = preprocess(batch)
    loss, logits = loss_fn(model, manifold_inputs, labels)
    metrics.update(loss=loss.detach().numpy(), logits=logits.detach().numpy(), labels=labels.detach().numpy())


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
        train_step(model, optim, metrics, batch)


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

# weights grads: tensor([[ 0.0113, -0.0020, -0.0075, -0.0120, -0.0014,  0.0063,  0.0056,  0.0063,                                                  | 0/1875 [00:00<?, ?it/s]
#           0.0042,  0.0020],
#         [ 0.0036, -0.0050, -0.0072, -0.0118, -0.0010,  0.0060,  0.0054,  0.0063,
#           0.0041,  0.0021],
#         [ 0.0037, -0.0019, -0.0257, -0.0123, -0.0017,  0.0058,  0.0055,  0.0063,
#           0.0040,  0.0021],
#         [ 0.0034, -0.0022, -0.0071, -0.0408, -0.0015,  0.0062,  0.0055,  0.0063,
#           0.0041,  0.0019],
#         [ 0.0033, -0.0017, -0.0072, -0.0111, -0.0035,  0.0057,  0.0052,  0.0060,
#           0.0037,  0.0020],
#         [ 0.0031, -0.0024, -0.0076, -0.0117, -0.0019,  0.0192,  0.0051,  0.0057,
#           0.0033,  0.0019],
#         [ 0.0032, -0.0026, -0.0080, -0.0125, -0.0020,  0.0057,  0.0174,  0.0060,
#           0.0038,  0.0022],
#         [ 0.0029, -0.0021, -0.0082, -0.0122, -0.0016,  0.0052,  0.0048,  0.0192,
#           0.0031,  0.0021],
#         [ 0.0028, -0.0028, -0.0086, -0.0124, -0.0021,  0.0053,  0.0049,  0.0053,
#           0.0088,  0.0020]]),
# bias grads: tensor([-4.6478e-02,  9.0341e-02,  3.8723e-02,  7.8561e-02,  2.7807e-02,
#         -4.9353e-02, -4.8446e-02, -4.9340e-02, -4.4863e-02, -9.2744e-18])

# TORCH LOGITS tensor([[ 2.5578e-05,  1.6859e-04,  3.3180e-04,  3.3947e-04, -3.6960e-04,