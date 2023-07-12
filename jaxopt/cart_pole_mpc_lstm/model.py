from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax import linen as nn
import brax
from brax import base
from brax.positional import pipeline
import distrax

PRNGKey = Any

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


class ActorCriticNetwork(nn.Module):
    action_space: int
    nodes: int
    pipeline_state: base.State

    def setup(self):
        features = 128
        self.mpc_layer_1 = nn.Dense(
            features=features,
            name='mpc_layer_1',
            dtype=dtype,
        )
        self.mpc_layer_2 = nn.Dense(
            features=features,
            name='mpc_layer_2',
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
        init_pipeline = lambda x, i: pipeline.init(self.pipeline_state, x, i)
        self.init_pipeline = jax.jit(init_pipeline)
        step_pipeline = lambda x, i: pipeline.step(self.pipeline_state, x, i)
        self.step_pipeline = jax.jit(step_pipeline)

    def model(self, x, key):
        def MPCCell(
            carry: Tuple[jax.Array, base.State, PRNGKey],
            xs: None,
        ) -> Tuple[Tuple[jnp.ndarray, base.State, PRNGKey], jnp.ndarray]:
            x, state, key = carry
            x = self.mpc_layer_1(x)
            x = nn.tanh(x)
            x = self.mpc_layer_2(x)
            x_mu = nn.tanh(x[0])
            x_std = nn.sigmoid(x[1])
            probability_distribution = distrax.Normal(loc=x_mu, scale=x_std)
            x = probability_distribution.sample(seed=key)
            state = self.step_pipeline(state, x)
            x = jnp.concatenate([state.q, state.qd])
            _, key = jax.random.split(key)
            carry = x, state, key
            data = x
            return carry, data

        # Create Initial Pipeline State:
        q = x[:2]
        qd = x[2:]
        state = self.init_pipeline(q, qd)

        # Scan MPC Block:
        carry, data = jax.lax.scan(
            f=MPCCell,
            init=(x, state, key),
            xs=None,
            length=self.nodes,
        )
        x = data
        # Trajectory:
        state_trajectory = x

        # Mean Layer:
        i = self.dense_1(state_trajectory)
        i = nn.tanh(i)
        i = self.dense_2(i)
        i = nn.tanh(i)
        # Standard Deviation Layer:
        j = self.dense_3(state_trajectory)
        j = nn.tanh(j)
        j = self.dense_4(j)
        j = nn.tanh(j)
        # Value Layer:
        k = self.dense_5(state_trajectory)
        k = nn.tanh(k)
        k = self.dense_6(k)
        k = nn.tanh(k)

        # Output Layer:
        mean = self.mean_layer(i)
        mean = nn.tanh(mean)
        std = self.std_layer(j)
        std = nn.sigmoid(std)
        values = self.value_layer(k)
        return mean, std, values

    def __call__(self, x, key):
        mean, std, values = self.model(x, key)
        return mean, std, values


class ActorCriticNetworkVmap(nn.Module):
    action_space: int
    nodes: int
    pipeline_state: base.State

    def setup(self) -> None:
        # Shared Params:
        self.model = nn.vmap(
            ActorCriticNetwork,
            variable_axes={'params': None},
            split_rngs={'params': False},
            in_axes=0,
        )(
            action_space=self.action_space,
            nodes=self.nodes,
            pipeline_state=self.pipeline_state,
        )

    def __call__(self, x, key):
        mean, std, values = self.model(x, key)
        return mean, std, values
