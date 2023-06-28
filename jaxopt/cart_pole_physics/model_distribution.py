import jax
import jax.numpy as jnp
from flax import linen as nn
import brax
from brax.positional import pipeline
import distrax

jax.config.update("jax_enable_x64", True)


class ActorCriticNetwork(nn.Module):
    action_space: int
    env: brax.positional.base.State

    def setup(self):
        dtype = jnp.float64
        features = 128
        self.pipeline_layer_1 = nn.Dense(
            features=features,
            name='pipeline_layer_1',
            dtype=dtype,
        )
        self.pipeline_layer_3 = nn.Dense(
            features=features,
            name='pipeline_layer_3',
            dtype=dtype,
        )
        self.pipeline_layer_5 = nn.Dense(
            features=features,
            name='pipeline_layer_5',
            dtype=dtype,
        )
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
        self.pipeline_layer_2 = nn.Dense(
            features=2,
            name='pipeline_layer_2',
            dtype=dtype,
        )
        self.pipeline_layer_4 = nn.Dense(
            features=2,
            name='pipeline_layer_4',
            dtype=dtype,
        )
        self.pipeline_layer_6 = nn.Dense(
            features=2,
            name='pipeline_layer_6',
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
        init_pipeline = lambda x, y: pipeline.init(self.env, x, y)
        self.init_pipeline = jax.jit(init_pipeline)
        step_pipeline = lambda x, y: pipeline.step(self.env, x, y)
        self.step_pipeline = jax.jit(step_pipeline)

    # Small Network:
    def model(self, x, key):
        # Limit Output Range:
        range_limit = 0.5

        # Create Initial Pipeline State:
        q = x[:2]
        qd = x[2:]
        state = self.init_pipeline(q, qd)

        # Pipeline Layer 1:
        y = self.pipeline_layer_1(x)
        y = nn.tanh(y)
        # Output Mean and Std:
        y = self.pipeline_layer_2(y)
        y_mu = range_limit * nn.tanh(y[0])
        y_std = range_limit * nn.sigmoid(y[1])
        # Sample Action:
        probability_distribution = distrax.Normal(loc=y_mu, scale=y_std)
        y = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, y)
        state_1 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 2:
        z = self.pipeline_layer_3(state_1)
        z = nn.tanh(z)
        z = self.pipeline_layer_4(z)
        z_mu = range_limit * nn.tanh(z[0])
        z_std = range_limit * nn.sigmoid(z[1])
        probability_distribution = distrax.Normal(loc=z_mu, scale=z_std)
        z = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, z)
        state_2 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 3:
        i = self.pipeline_layer_5(state_2)
        i = nn.tanh(i)
        i = self.pipeline_layer_6(i)
        i_mu = range_limit * nn.tanh(i[0])
        i_std = range_limit * nn.sigmoid(i[1])
        probability_distribution = distrax.Normal(loc=i_mu, scale=i_std)
        i = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, i)
        state_3 = jnp.concatenate([state.q, state.qd])

        # Trajectory:
        state_trajectory = jnp.concatenate([state_1, state_2, state_3])

        # Mean Layer:
        j = self.dense_1(state_trajectory)
        j = nn.tanh(j)
        j = self.dense_2(j)
        j = nn.tanh(j)
        # Standard Deviation Layer:
        k = self.dense_3(state_trajectory)
        k = nn.tanh(k)
        k = self.dense_4(k)
        k = nn.tanh(k)
        # Value Layer:
        w = self.dense_5(state_trajectory)
        w = nn.tanh(w)
        w = self.dense_6(w)
        w = nn.tanh(w)

        # Output Layer:
        mean = self.mean_layer(j)
        mean = range_limit * nn.tanh(mean)
        std = self.std_layer(k)
        std = range_limit * nn.sigmoid(std)
        values = self.value_layer(w)
        return mean, std, values

    def __call__(self, x, key):
        mean, std, values = self.model(x, key)
        return mean, std, values


class ActorCriticNetworkVmap(nn.Module):
    action_space: int
    env: brax.generalized.base.State

    def setup(self) -> None:
        # Shared Params:
        self.model = nn.vmap(
            ActorCriticNetwork,
            variable_axes={'params': None},
            split_rngs={'params': False},
            in_axes=0,
        )(
            action_space=self.action_space,
            env=self.env,
        )

    def __call__(self, x, key):
        mean, std, values = self.model(x, key)
        return mean, std, values
