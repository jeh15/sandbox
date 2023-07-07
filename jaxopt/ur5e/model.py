import jax
import jax.numpy as jnp
from flax import linen as nn
import brax
from brax.positional import pipeline
import distrax

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


class ActorCriticNetwork(nn.Module):
    action_space: int
    env: brax.positional.base.State

    def setup(self):
        features = 128
        self.pipeline_layer_1 = nn.Dense(
            features=features,
            name='pipeline_layer_1',
            dtype=dtype,
        )
        self.pipeline_layer_2 = nn.Dense(
            features=features,
            name='pipeline_layer_2',
            dtype=dtype,
        )
        self.pipeline_layer_3 = nn.Dense(
            features=features,
            name='pipeline_layer_3',
            dtype=dtype,
        )
        self.pipeline_layer_4 = nn.Dense(
            features=features,
            name='pipeline_layer_4',
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
        self.pipeline_layer_1_mean = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_1_mean',
            dtype=dtype,
        )
        self.pipeline_layer_1_std = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_std',
            dtype=dtype,
        )
        self.pipeline_layer_2_mean = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_2_mean',
            dtype=dtype,
        )
        self.pipeline_layer_2_std = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_2_std',
            dtype=dtype,
        )
        self.pipeline_layer_3_mean = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_3_mean',
            dtype=dtype,
        )
        self.pipeline_layer_3_std = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_3_std',
            dtype=dtype,
        )
        self.pipeline_layer_4_mean = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_4_mean',
            dtype=dtype,
        )
        self.pipeline_layer_4_std = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_4_std',
            dtype=dtype,
        )
        self.pipeline_layer_5_mean = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_5_mean',
            dtype=dtype,
        )
        self.pipeline_layer_5_std = nn.Dense(
            features=self.action_space,
            name='pipeline_layer_5_std',
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

    def model(self, x, key):
        # Limit Output Range:
        high_limit = 150.0 / 2.0
        low_limit = 28.0 / 2.0
        range_limit = jnp.array(
            [
                high_limit, high_limit, high_limit,
                low_limit, low_limit, low_limit,
            ],
            dtype=dtype,
        )

        # Create Initial Pipeline State:
        q = x[:6]
        qd = x[6:]
        state = self.init_pipeline(q, qd)

        # Pipeline Layer 1:
        x = self.pipeline_layer_1(x)
        x = nn.tanh(x)
        x_mu = self.pipeline_layer_1_mean(x)
        x_std = self.pipeline_layer_1_std(x)
        x_mu = range_limit * nn.tanh(x_mu)
        x_std = nn.sigmoid(x_std)
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=x_mu,
            scale_diag=x_std,
        )
        x = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, x)
        state_1 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 2:
        x = self.pipeline_layer_2(state_1)
        x = nn.tanh(x)
        x_mu = self.pipeline_layer_2_mean(x)
        x_std = self.pipeline_layer_2_std(x)
        x_mu = range_limit * nn.tanh(x_mu)
        x_std = nn.sigmoid(x_std)
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=x_mu,
            scale_diag=x_std,
        )
        x = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, x)
        state_2 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 3:
        x = self.pipeline_layer_3(state_2)
        x = nn.tanh(x)
        x_mu = self.pipeline_layer_3_mean(x)
        x_std = self.pipeline_layer_3_std(x)
        x_mu = range_limit * nn.tanh(x_mu)
        x_std = nn.sigmoid(x_std)
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=x_mu,
            scale_diag=x_std,
        )
        x = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, x)
        state_3 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 4:
        x = self.pipeline_layer_4(state_3)
        x = nn.tanh(x)
        x_mu = self.pipeline_layer_4_mean(x)
        x_std = self.pipeline_layer_4_std(x)
        x_mu = range_limit * nn.tanh(x_mu)
        x_std = nn.sigmoid(x_std)
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=x_mu,
            scale_diag=x_std,
        )
        x = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, x)
        state_4 = jnp.concatenate([state.q, state.qd])

        # Pipeline Layer 5:
        x = self.pipeline_layer_5(state_4)
        x = nn.tanh(x)
        x_mu = self.pipeline_layer_5_mean(x)
        x_std = self.pipeline_layer_5_std(x)
        x_mu = range_limit * nn.tanh(x_mu)
        x_std = nn.sigmoid(x_std)
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=x_mu,
            scale_diag=x_std,
        )
        x = probability_distribution.sample(seed=key)
        state = self.step_pipeline(state, x)
        state_5 = jnp.concatenate([state.q, state.qd])

        # Trajectory:
        state_trajectory = jnp.concatenate(
            [
                state_1, state_2, state_3, state_4, state_5,
            ],
        )

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
