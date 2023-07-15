from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from brax import base
from brax.positional import pipeline
import distrax

PRNGKey = Any

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

range_limit = 0.5


class _MPCCell(nn.Module):
    """
        Module for scan function for the MPC cell.
    """
    features: int
    pipeline_state: base.State

    def setup(self) -> None:
        self.shared_layer_1 = nn.Dense(
            features=self.features,
            name='shared_layer_1',
            dtype=dtype,
        )
        self.shared_layer_2 = nn.Dense(
            features=self.features,
            name='shared_layer_2',
            dtype=dtype,
        )
        self.policy_mean_1 = nn.Dense(
            features=self.features,
            name='policy_mean_1',
            dtype=dtype,
        )
        self.policy_mean_2 = nn.Dense(
            features=1,
            name='policy_mean_2',
            dtype=dtype,
        )
        self.policy_std_1 = nn.Dense(
            features=self.features,
            name='policy_std_1',
            dtype=dtype,
        )
        self.policy_std_2 = nn.Dense(
            features=1,
            name='policy_std_2',
            dtype=dtype,
        )
        step_pipeline = lambda x, i: pipeline.step(self.pipeline_state, x, i)
        self.step_pipeline = jax.jit(step_pipeline)

    def __call__(
        self,
        carry: Tuple[jax.Array, base.State, PRNGKey],
        xs: None,
    ) -> Tuple[Tuple[jnp.ndarray, base.State, PRNGKey], jnp.ndarray]:
        x, state, key = carry
        x = self.shared_layer_1(x)
        x = nn.tanh(x)
        x = self.shared_layer_2(x)
        x = nn.tanh(x)
        x_mu = self.policy_mean_1(x)
        x_mu = nn.tanh(x_mu)
        x_mu = self.policy_mean_2(x_mu)
        x_mu = range_limit * nn.tanh(x_mu)
        x_std = self.policy_std_1(x)
        x_std = nn.sigmoid(x_std)
        x_std = self.policy_std_2(x_std)
        x_std = nn.sigmoid(x_std)
        probability_distribution = distrax.Normal(loc=x_mu, scale=x_std)
        x = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, x)
        x = jnp.concatenate([state.q, state.qd])
        _, key = jax.random.split(key)
        carry = x, state, key
        data = x
        return carry, data


class MPCCell(nn.Module):
    features: int
    nodes: int
    pipeline_state: base.State

    def setup(self):
        # Initialize Pipeline:
        init_pipeline = lambda x, i: pipeline.init(self.pipeline_state, x, i)
        self.init_pipeline = jax.jit(init_pipeline)
        # Create Scan Function:
        scan_fn = nn.scan(
            _MPCCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=0,
            out_axes=0,
            length=self.nodes,
        )
        self.scan = scan_fn(self.features, self.pipeline_state)

    def __call__(self, x, key) -> jnp.ndarray:
        # Initialize Pipeline State:
        q = x[:2]
        qd = x[2:]
        state = self.init_pipeline(q, qd)
        initial_condition = jnp.concatenate([state.q, state.qd])

        # Run Scan Function:
        carry = x, state, key
        carry, data = self.scan(carry, None)

        # Concatenate State Trajectory:
        predicted_state_trajectory = jnp.concatenate(data)
        state_trajectory = jnp.concatenate(
            [initial_condition, predicted_state_trajectory],
        )
        return state_trajectory


class ActorCriticNetwork(nn.Module):
    action_space: int
    nodes: int
    pipeline_state: base.State

    def setup(self):
        features = 128
        MPC_features = 128
        self.shared_layer_1 = nn.Dense(
            features=features,
            name='shared_layer_1',
            dtype=dtype,
        )
        self.shared_layer_2 = nn.Dense(
            features=features,
            name='shared_layer_2',
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
        self.MPCCell = MPCCell(
            features=MPC_features,
            nodes=self.nodes,
            pipeline_state=self.pipeline_state,
        )

    def model(self, x, key):
        """
            What I changed...
            state_trajectory now includes the initial condition.
            Added a shared layer before the mean, std, and value layers.
        """
        # Run MPC Block:
        state_trajectory = self.MPCCell(x, key)

        # Shared Layer:
        x = self.shared_layer_1(state_trajectory)
        x = nn.tanh(x)
        x = self.shared_layer_2(x)
        x = nn.tanh(x)

        # Mean Layer:
        i = self.dense_1(x)
        i = nn.tanh(i)
        i = self.dense_2(i)
        i = nn.tanh(i)
        # Standard Deviation Layer:
        j = self.dense_3(x)
        j = nn.tanh(j)
        j = self.dense_4(j)
        j = nn.tanh(j)
        # Value Layer:
        k = self.dense_5(x)
        k = nn.tanh(k)
        k = self.dense_6(k)
        k = nn.tanh(k)

        # Output Layer:
        mean = self.mean_layer(i)
        mean = range_limit * nn.tanh(mean)
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
