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
            features=self.action_space,
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

    def __call__(self, x):
        logits = self.actor(x)
        policy_probabilities = nn.softmax(logits)
        value = self.critic(x)
        return policy_probabilities, value

    def policy_probabilities(self, x):
        logits = self.actor(x)
        policy_probabilities = nn.softmax(logits)
        return policy_probabilities

    def value(self, x):
        return self.critic(x)

    def policy_action(self, x):
        logits = self.actor(x)
        policy_probabilities = nn.softmax(logits)
        distribution = distrax.Categorical(probs=policy_probabilities)
        return distribution.sample(seed=self.seed)
