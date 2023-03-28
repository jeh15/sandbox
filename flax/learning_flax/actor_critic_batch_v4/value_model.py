import jax.numpy as jnp
from flax import linen as nn


class ValueNetwork(nn.Module):
    def setup(self):
        dtype = jnp.float32
        self.dense_1 = nn.Dense(
            features=64,
            name='layer_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=64,
            name='layer_2',
            dtype=dtype,
        )
        self.value_layer = nn.Dense(
            features=1,
            name='value_layer',
            dtype=dtype,
        )

    def model(self, x):
        x = self.dense_1(x)
        x = nn.relu(x)
        x = self.dense_2(x)
        x = nn.relu(x)
        values = self.value_layer(x)
        return values

    def __call__(self, x):
        values = self.model(x)
        return values
