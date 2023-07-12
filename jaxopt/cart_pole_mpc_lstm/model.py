import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
import brax
from brax.positional import pipeline
import distrax

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


class _MPCCell(nn.Module):
    hidden_size: int
    env: brax.positional.base.State

    def setup(self) -> None:
        self.dense_1 = nn.Dense(
            features=self.hidden_size,
            name='dense_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=2,
            name='dense_2',
            dtype=dtype,
        )
        step_pipeline = lambda x, i: pipeline.step(self.env, x, i)
        self.step_pipeline = jax.jit(step_pipeline)

    def __call__(
        self: Self,
        carry: Tuple[State, PRNGKey],
        x: jax.typing.Array,
    ) -> Tuple[Tuple[State, PRNGKey], jax.typing.Array]:
        state, key = carry
        range_limit = 0.5
        x = self.dense_1(x)
        x = nn.tanh(x)
        x = self.dense_2(x)
        x_mu = range_limit * nn.tanh(x[0])
        x_std = nn.sigmoid(x[1])
        probability_distribution = distrax.Normal(loc=x_mu, scale=x_std)
        x = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, x)
        x = jnp.concatenate([state.q, state.qd])
        _, key = jax.random.split(key)
        carry = state, key
        data = x
        return carry, data


class MPCCell(nn.Module):
    hidden_size: int
    env: brax.positional.base.State
    nodes: int

    def setup(self: Self) -> None:
        init_pipeline = lambda x, i: pipeline.init(self.env, x, i)
        self.init_pipeline = jax.jit(init_pipeline)

    def __call__(self, x, key):
        # Create initial state:
        q = x[:2]
        qd = x[2:]
        state = self.init_pipeline(q, qd)

        # Scan MPC Block:
        scanMPC = nn.scan(
            _MPCCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1,
            length=self.nodes,
        )
        carry = state, key
        carry, x = scanMPC(carry, x)

        return x


class __MPCBlock__(nn.Module):
    """MPC Layer"""
    hidden_size: int
    env: brax.positional.base.State
    nodes: int

    def setup(self) -> None:
        self.dense_1 = nn.Dense(
            features=self.hidden_size,
            name='dense_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=2,
            name='dense_2',
            dtype=dtype,
        )
        init_pipeline = lambda x, i: pipeline.init(self.env, x, i)
        self.init_pipeline = jax.jit(init_pipeline)
        step_pipeline = lambda x, i: pipeline.step(self.env, x, i)
        self.step_pipeline = jax.jit(step_pipeline)

    def __call__(self, carry, x):
        # Scan Function:
        def MPCCell(carry, x):
            state, key = carry
            range_limit = 0.5
            x = self.dense_1(x)
            x = nn.tanh(x)
            x = self.dense_2(x)
            x_mu = range_limit * nn.tanh(x[0])
            x_std = nn.sigmoid(x[1])
            probability_distribution = distrax.Normal(loc=x_mu, scale=x_std)
            x = probability_distribution.sample(seed=key)
            state = self.step_pipeline(state, x)
            x = jnp.concatenate([state.q, state.qd])
            _, key = jax.random.split(key)
            carry = state, key
            data = x
            return carry, data

        # Create initial state:
        q = x[:2]
        qd = x[2:]
        state = self.init_pipeline(q, qd)

        # Scan MPC Block:
        scan = nn.scan(
            MPCCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=0,
            out_axes=0,
        )

        return x


class ActorCriticNetwork(nn.Module):
    action_space: int
    env: brax.positional.base.State

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
        init_pipeline = lambda x, i: pipeline.init(self.env, x, i)
        self.init_pipeline = jax.jit(init_pipeline)
        step_pipeline = lambda x, i: pipeline.step(self.env, x, i)
        self.step_pipeline = jax.jit(step_pipeline)

    def model(self, x, key):
        def MPCCell(
            carry: Tuple[jax.typing.Array, State, PRNGKey],
            _: None,
        ) -> Tuple[Tuple[jnp.ndarray, State, PRNGKey], jnp.ndarray]:
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
    env: brax.positional.base.State

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
