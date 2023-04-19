from typing import Any, Callable, Sequence

import jax
from jax import numpy as jnp

from flax import linen as nn
from flax.training import train_state

import numpy as np

import optax

import ml_collections
from ml_collections import config_flags

import tensorflow as tf
import tensorflow_datasets as tfds

# Needed to use google flags (TODO: Look into)
from absl import app
from absl import flags

import pdb


# Configuration Setup:
def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 128
    config.num_epochs = 10
    return config


# Flax CNN:
class CNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


# Helper Functions:
@jax.jit
def apply_model(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Returns:
        The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state,
            train_ds,
            config.batch_size,
            input_rng,
            )

        _, test_loss, test_accuracy = apply_model(
            state,
            test_ds['image'],
            test_ds['label']
            )

        # Logging:
        print(f"Epoch: {epoch}, train_loss: {train_loss}, train_accuracy: {train_accuracy * 100}, test_loss: {test_loss}, test_accuracy: {test_accuracy * 100}")

    return state


# Main:
def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    config = get_config()
    train_and_evaluate(config)


if __name__ == '__main__':
    app.run(main)
