from typing import Any, Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax


class ActorCritic(nn.Module):
    action_space: int

    # runs after __postinit__:
    def setup(self):
        dtype = jnp.float32
        # Actor Network:
        self._actor_dense_1 = nn.Dense(
            features=64,
            name='actor_layer_1',
            dtype=dtype,
        )
        self._actor_dense_2 = nn.Dense(
            features=64,
            name='actor_layer_2',
            dtype=dtype,
        )
        self._actor_dense_3 = nn.Dense(
            features=self.action_space,
            name='logits_layer',
            dtype=dtype,
        )
        # Critic Network:
        self._critic_dense_1 = nn.Dense(
            features=64,
            name='critic_layer_1',
            dtype=dtype,
        )
        self._critic_dense_2 = nn.Dense(
            features=64,
            name='critic_layer_2',
            dtype=dtype,
        )
        self._critic_dense_3 = nn.Dense(
            features=1,
            name='value_layer',
            dtype=dtype,
        )

    def actor(self, x):
        x = self._actor_dense_1(x)
        x = nn.relu(x)
        x = self._actor_dense_2(x)
        x = nn.relu(x)
        logits = self._actor_dense_3(x)
        return logits

    def critic(self, x):
        x = self._critic_dense_1(x)
        x = nn.relu(x)
        x = self._critic_dense_2(x)
        x = nn.relu(x)
        value = self._critic_dense_3(x)
        return value

    def __call__(self, x):
        logits = self.actor(x)
        policy_probabilities = nn.softmax(logits)
        value = self.critic(x)
        return policy_probabilities, value
