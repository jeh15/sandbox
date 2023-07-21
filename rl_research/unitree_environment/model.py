from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from brax import base
from brax.base import System
from brax.positional import pipeline
import distrax

PRNGKey = jax.random.PRNGKeyArray

# jax.config.update("jax_enable_x64", True)
dtype = jnp.float32


class _MPCCell(nn.Module):
    """
        Module for scan function for the MPC cell.
    """
    features: int
    sys: System

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
        step_pipeline = lambda state, action: pipeline.step(self.sys, state, action)
        self.step_pipeline = jax.jit(step_pipeline)

    def __call__(
        self,
        carry: Tuple[jax.Array, base.State, PRNGKey],
        xs: None,
    ) -> Tuple[Tuple[jnp.ndarray, base.State, PRNGKey], jnp.ndarray]:
        x, state, key = carry
        x = self.shared_layer_1(x)
        x = nn.relu(x)
        x = self.shared_layer_2(x)
        x = nn.relu(x)
        x_mu = self.policy_mean_1(x)
        x_mu = nn.relu(x_mu)
        x_mu = self.policy_mean_2(x_mu)
        x_std = self.policy_std_1(x)
        x_std = nn.relu(x_std)
        x_std = self.policy_std_2(x_std)
        x_std = nn.relu(x_std)
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=x_mu,
            scale_diag=x_std,
        )
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
    sys: System

    def setup(self):
        # Initialize Pipeline:
        init_pipeline = lambda q, qd: pipeline.init(self.sys, q, qd)
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
        self.scan = scan_fn(self.features, self.sys)

    def __call__(self, x, key) -> jnp.ndarray:
        # Initialize Pipeline State:
        q = x[:self.sys.init_q.shape[0]]
        qd = x[self.sys.init_q.shape[0]:]
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
    sys: System

    def setup(self):
        features = 64
        MPC_features = 64
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
            sys=self.sys,
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
        x = nn.relu(x)
        x = self.shared_layer_2(x)
        x = nn.relu(x)

        # Mean Layer:
        i = self.dense_1(x)
        i = nn.relu(i)
        i = self.dense_2(i)
        i = nn.relu(i)
        # Standard Deviation Layer:
        j = self.dense_3(x)
        j = nn.relu(j)
        j = self.dense_4(j)
        j = nn.relu(j)
        # Value Layer:
        k = self.dense_5(x)
        k = nn.relu(k)
        k = self.dense_6(k)
        k = nn.relu(k)

        # Output Layer:
        mean = self.mean_layer(i)
        std = self.std_layer(j)
        std = nn.relu(std)
        values = self.value_layer(k)
        return mean, std, values

    def __call__(self, x, key):
        mean, std, values = self.model(x, key)
        return mean, std, values


class ActorCriticNetworkVmap(nn.Module):
    action_space: int
    nodes: int
    sys: System

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
            sys=self.sys,
        )

    def __call__(self, x, key):
        mean, std, values = self.model(x, key)
        return mean, std, values
