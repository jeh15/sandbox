from typing import Any, Callable, Sequence

import jax
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn

import optax

model = nn.Dense(features=5)

key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (10,))  # Dummy input data
params = model.init(key2, x ) # Initialization call

model.apply(params, x)

# Set problem dimensions.
n_samples = 20
x_dim = 10
y_dim = 5

# Generate random ground truth W and b.
key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))
# Store the parameters in a FrozenDict pytree.
true_params = freeze({'params': {'bias': b, 'kernel': W}})

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(key_noise, (n_samples, y_dim))
print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)


# Same as JAX version but using model.apply().
@jax.jit
def mse(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y-pred, y-pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W, b: ', mse(true_params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(mse)


tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
print(opt_state)
loss_grad_fn = jax.value_and_grad(mse)


for i in range(101):
    # Perform one gradient update.
    loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print(f'Loss step {i}: ', loss_val)

