import jax.numpy as jnp
from flax import linen as nn


class CriticNetwork(nn.Module):
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
            features=1,
            name='output_layer',
            dtype=dtype,
        )

    def critic(self, x):
        x = self.dense_1(x)
        x = nn.relu(x)
        x = self.dense_2(x)
        x = nn.relu(x)
        value = self.output_layer(x)
        return value

    def __call__(self, x):
        value = self.critic(x)
        return value
