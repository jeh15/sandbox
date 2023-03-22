import jax.numpy as jnp
from flax import linen as nn


class ActorNetwork(nn.Module):
    action_space: int

    def setup(self):
        dtype = jnp.float32
        self.dense_1 = nn.Dense(
            features=2*64,
            name='layer_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=2*64,
            name='layer_2',
            dtype=dtype,
        )
        self.output_layer = nn.Dense(
            features=self.action_space,
            name='output_layer',
            dtype=dtype,
        )

    def actor(self, x):
        x = self.dense_1(x)
        x = nn.relu(x)
        x = self.dense_2(x)
        x = nn.relu(x)
        logits = self.output_layer(x)
        return logits

    def __call__(self, x):
        logits = self.actor(x)
        return logits
