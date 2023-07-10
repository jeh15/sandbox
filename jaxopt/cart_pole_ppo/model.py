import jax
import jax.numpy as jnp
from flax import linen as nn

jax.config.update("jax_enable_x64", True)


class ActorCriticNetwork(nn.Module):
    action_space: int

    def setup(self):
        dtype = jnp.float64
        features = 128
        self.dense_1 = nn.Dense(
            features=features,
            name='dense_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=features,
            name='dense_2',
            dtype=dtype,
        )
        self.dense_3 = nn.Dense(
            features=features,
            name='dense_3',
            dtype=dtype,
        )
        self.dense_4 = nn.Dense(
            features=features,
            name='dense_4',
            dtype=dtype,
        )
        self.dense_5 = nn.Dense(
            features=features,
            name='dense_5',
            dtype=dtype,
        )
        self.dense_6 = nn.Dense(
            features=features,
            name='dense_6',
            dtype=dtype,
        )
        self.dense_7 = nn.Dense(
            features=features,
            name='dense_7',
            dtype=dtype,
        )
        self.dense_8 = nn.Dense(
            features=features,
            name='dense_8',
            dtype=dtype,
        )
        self.dense_9 = nn.Dense(
            features=features,
            name='dense_9',
            dtype=dtype,
        )
        self.dense_10 = nn.Dense(
            features=features,
            name='dense_10',
            dtype=dtype,
        )
        self.dense_11 = nn.Dense(
            features=features,
            name='dense_11',
            dtype=dtype,
        )
        self.dense_12 = nn.Dense(
            features=features,
            name='dense_12',
            dtype=dtype,
        )
        self.dense_13 = nn.Dense(
            features=features,
            name='dense_13',
            dtype=dtype,
        )
        self.dense_14 = nn.Dense(
            features=features,
            name='dense_14',
            dtype=dtype,
        )
        self.dense_15 = nn.Dense(
            features=features,
            name='dense_15',
            dtype=dtype,
        )
        self.dense_16 = nn.Dense(
            features=features,
            name='dense_16',
            dtype=dtype,
        ) 
        self.mean_layer = nn.Dense(
            features=self.action_space,
            name='mean_layer',
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

    # Small Network:
    def model(self, x):
        # Limit Output Range:
        range_limit = 0.5

        x = self.dense_1(x)
        x = nn.tanh(x)
        x = self.dense_2(x)
        x = nn.tanh(x)
        x = self.dense_3(x)
        x = nn.tanh(x)
        x = self.dense_4(x)
        x = nn.tanh(x)
        x = self.dense_5(x)
        x = nn.tanh(x)
        x = self.dense_6(x)
        x = nn.tanh(x)
        x = self.dense_7(x)
        x = nn.tanh(x)
        x = self.dense_8(x)
        x = nn.tanh(x)
        x = self.dense_9(x)
        x = nn.tanh(x)
        x = self.dense_10(x)
        x = nn.tanh(x)

        # Policy Layer: Nonlinear Function of Acceleration
        y = self.dense_1(x)
        y = nn.tanh(y)
        y = self.dense_2(y)
        y = nn.tanh(y)
        # Pipeline that decides std should have more information of the states
        z = self.dense_3(x)
        z = nn.tanh(z)
        z = self.dense_4(z)
        z = nn.tanh(z)

        # Value Layer: Nonlinear Function of Objective Value
        w = self.dense_5(x)
        w = nn.tanh(w)
        w = self.dense_6(w)
        w = nn.tanh(w)

        # Output Layer: (changed from softmax to sigmoid)
        mean = self.mean_layer(y)
        mean = range_limit * nn.tanh(mean)
        std = self.std_layer(z)
        std = nn.sigmoid(std)
        values = self.value_layer(w)
        return mean, std, values

    def __call__(self, x):
        mean, std, values = self.model(x)
        return mean, std, values


class ActorCriticNetworkVmap(nn.Module):
    action_space: int

    def setup(self) -> None:
        # Shared Params:
        self.model = nn.vmap(
            ActorCriticNetwork,
            variable_axes={'params': None},
            split_rngs={'params': False},
            in_axes=0,
        )(
            action_space=self.action_space,
        )

    def __call__(self, x):
        mean, std, values = self.model(x)
        return mean, std, values
