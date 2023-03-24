import jax.numpy as jnp
from flax import linen as nn


class ActorCriticNetwork(nn.Module):
    action_space: int

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
        self.dense_3 = nn.Dense(
            features=64,
            name='layer_3',
            dtype=dtype,
        )
        self.dense_4 = nn.Dense(
            features=64,
            name='layer_4',
            dtype=dtype,
        )
        self.logit_layer = nn.Dense(
            features=self.action_space,
            name='logit_layer',
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
        x = self.dense_3(x)
        x = nn.relu(x)
        y = self.dense_4(x)
        y = nn.relu(y)
        logits = self.logit_layer(x)
        values = self.value_layer(y)
        return logits, values

    def __call__(self, x):
        logits, values = self.model(x)
        return logits, values
