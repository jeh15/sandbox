import jax
import jax.numpy as jnp
from flax import linen as nn

import qp


class ActorCriticNetwork(nn.Module):
    action_space: int
    time_horizon: float
    nodes: int

    def setup(self):
        dtype = jnp.float32
        features = 64
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
        # Setup QP:
        equaility_functions, inequality_functions, objective_functions = (
            qp.qp_preprocess(
                self.time_horizon,
                self.nodes,
            )
        )
        # Isolate Function:
        self.osqp_layer = lambda x, y: qp.qp_layer(
            x,
            y,
            equaility_functions,
            inequality_functions,
            objective_functions,
            self.nodes,
        )

    # Embedded MPC Actor-Critic Network:
    def model(self, x):
        range = 2.0
        # QP Layer Inputs:
        initial_conditions = x
        target_position = jnp.array([1.0])
        # Shared Layers: QP
        pos, vel, acc, status = self.osqp_layer(
            initial_conditions, target_position
        )
        state_trajectory = jnp.concatenate([pos, vel, acc], axis=0)
        # Policy Layer: Use acceleration to predict mean and std
        y = self.dense_1(acc)
        y = nn.tanh(y)
        z = self.dense_2(acc)
        z = nn.tanh(z)
        # Value Layer: Use state trajectory to predict value (We could also include objective function value...)
        w = self.dense_3(state_trajectory)
        w = nn.tanh(w)
        w = self.dense_4(w)
        w = nn.tanh(w)
        # Output Layer:
        mean = self.mean_layer(y)
        mean = range * nn.tanh(mean)
        std = self.std_layer(z)
        std = nn.softplus(std)  # std != 0
        values = self.value_layer(w)
        return mean, std, values, status

    def __call__(self, x):
        mean, std, values, status = self.model(x)
        return mean, std, values, status


class ActorCriticNetworkVmap(nn.Module):
    action_space: int
    time_horizon: float
    nodes: int

    def setup(self) -> None:
        # Shared Params:
        self.model = nn.vmap(
            ActorCriticNetwork,
            variable_axes={'params': None},
            split_rngs={'params': False},
            in_axes=0,
        )(
            action_space=self.action_space,
            time_horizon=self.time_horizon,
            nodes=self.nodes,
        )

    def __call__(self, x):
        mean, std, values, status = self.model(x)
        return mean, std, values, status
