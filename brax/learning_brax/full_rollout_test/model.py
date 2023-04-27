import jax.numpy as jnp
from flax import linen as nn


class ActorCriticNetwork(nn.Module):
    action_space: int

    def setup(self):
        dtype = jnp.float32
        features = 128
        self.dense_1 = nn.Dense(
            features=features,
            name='layer_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=features,
            name='layer_2',
            dtype=dtype,
        )
        self.dense_3 = nn.Dense(
            features=features,
            name='layer_3',
            dtype=dtype,
        )
        self.dense_4 = nn.Dense(
            features=features,
            name='layer_4',
            dtype=dtype,
        )
        self.dense_5 = nn.Dense(
            features=features,
            name='layer_5',
            dtype=dtype,
        )
        self.dense_6 = nn.Dense(
            features=features,
            name='layer_6',
            dtype=dtype,
        )
        self.dense_7 = nn.Dense(
            features=features,
            name='layer_7',
            dtype=dtype,
        )
        self.dense_8 = nn.Dense(
            features=features,
            name='layer_8',
            dtype=dtype,
        )
        self.mean_layer = nn.Dense(
            features=self.action_space,
            name='mean_layer',
            dtype=dtype,
        )
        self.std_layer = nn.Dense(
            features=self.action_space,
            name='std_layer',
            dtype=dtype,
        )
        self.value_layer = nn.Dense(
            features=1,
            name='value_layer',
            dtype=dtype,
        )

    def model(self, x):
        range = 2.0
        # Shared Layers:
        x = self.dense_1(x)
        x = nn.tanh(x)
        x = self.dense_2(x)
        x = nn.tanh(x)
        # Policy Layer:
        y = self.dense_3(x)
        y = nn.tanh(y)
        y = self.dense_4(y)
        y = nn.tanh(y)
        z = self.dense_5(x)
        z = nn.tanh(z)
        z = self.dense_6(z)
        z = nn.tanh(z)
        # Value Layer
        w = self.dense_7(x)
        w = nn.tanh(w)
        w = self.dense_8(w)
        w = nn.tanh(w)
        # Output Layer:
        mean = self.mean_layer(y)
        mean = range * nn.tanh(mean)
        std = self.std_layer(z)
        std = nn.softplus(std)  # std != 0
        values = self.value_layer(w)
        return mean, std, values

    def __call__(self, x):
        mean, std, values = self.model(x)
        return mean, std, values
