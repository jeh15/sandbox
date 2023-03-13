from typing import Any, Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn


class ActorCritic(nn.Module):
    actor_layers: Sequence[int]
    critic_layers: Sequence[int]
    num_outputs: int

    # runs after __postinit__:
    def setup(self):
        return NotImplementedError

    def actor(self, x):
        dtype = jnp.float32
        x = nn.Dense(
            features=64,
            name='layer_1',
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=64,
            name='layer_2',
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        logits = nn.Dense(
            features=self.num_outputs,
            name='logits',
            dtype=dtype,
        )(x)
        return logits

    def critic(self, x):
        dtype = jnp.float32
        x = nn.Dense(
            features=64,
            name='layer_1',
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=64,
            name='layer_2',
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        value = nn.Dense(
            features=1,
            name='value',
            dtype=dtype,
        )(x)
        return value
