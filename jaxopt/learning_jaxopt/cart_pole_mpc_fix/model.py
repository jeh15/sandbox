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
        self.pipeline_layer_7 = nn.Dense(
            features=features,
            name='pipeline_layer_7',
            dtype=dtype,
        )
        self.pipeline_layer_9 = nn.Dense(
            features=features,
            name='pipeline_layer_9',
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
        self.pipeline_layer_8 = nn.Dense(
            features=2,
            name='pipeline_layer_8',
            dtype=dtype,
        )
        self.pipeline_layer_10 = nn.Dense(
            features=2,
            name='pipeline_layer_10',
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

    # Small Network:
    def model(self, x, key):
        # Limit Output Range:
        range_limit = 0.5

        # Create Initial Pipeline State:
        q = x[:2]
        qd = x[2:]
        state = self.init_pipeline(q, qd)

        # Pipeline Layer 1:
        i = self.pipeline_layer_1(x)
        i = nn.tanh(i)
        # Output Mean and Std:
        i = self.pipeline_layer_2(i)
        i_mu = range_limit * nn.tanh(i[0])
        i_std = nn.sigmoid(i[1])
        # Sample Action:
        probability_distribution = distrax.Normal(loc=i_mu, scale=i_std)
        i = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, i)
        state_1 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 2:
        j = self.pipeline_layer_3(state_1)
        j = nn.tanh(j)
        j = self.pipeline_layer_4(j)
        j_mu = range_limit * nn.tanh(j[0])
        j_std = nn.sigmoid(j[1])
        probability_distribution = distrax.Normal(loc=j_mu, scale=j_std)
        j = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, j)
        state_2 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 3:
        k = self.pipeline_layer_5(state_2)
        k = nn.tanh(k)
        k = self.pipeline_layer_6(k)
        k_mu = range_limit * nn.tanh(k[0])
        k_std = nn.sigmoid(k[1])
        probability_distribution = distrax.Normal(loc=k_mu, scale=k_std)
        k = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, k)
        state_3 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 4:
        l = self.pipeline_layer_7(state_3)
        l = nn.tanh(l)
        l = self.pipeline_layer_8(l)
        l_mu = range_limit * nn.tanh(l[0])
        l_std = nn.sigmoid(l[1])
        probability_distribution = distrax.Normal(loc=l_mu, scale=l_std)
        l = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, l)
        state_4 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 5:
        m = self.pipeline_layer_9(state_4)
        m = nn.tanh(m)
        m = self.pipeline_layer_10(m)
        m_mu = range_limit * nn.tanh(m[0])
        m_std = nn.sigmoid(m[1])
        probability_distribution = distrax.Normal(loc=m_mu, scale=m_std)
        m = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, m)
        state_5 = jnp.concatenate([state.q, state.qd])

        # Trajectory:
        state_trajectory = jnp.concatenate(
            [state_1, state_2, state_3, state_4, state_5],
        )

        # Mean Layer:
        n = self.dense_1(state_trajectory)
        n = nn.tanh(n)
        n = self.dense_2(n)
        n = nn.tanh(n)
        # Standard Deviation Layer:
        o = self.dense_3(state_trajectory)
        o = nn.tanh(o)
        o = self.dense_4(o)
        o = nn.tanh(o)
        # Value Layer:
        p = self.dense_5(state_trajectory)
        p = nn.tanh(p)
        p = self.dense_6(p)
        p = nn.tanh(p)

        # Output Layer:
        mean = self.mean_layer(n)
        mean = range_limit * nn.tanh(mean)
        std = self.std_layer(o)
        std = nn.sigmoid(std)
        values = self.value_layer(p)
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
