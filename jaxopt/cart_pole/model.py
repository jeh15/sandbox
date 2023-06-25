import jax
import jax.numpy as jnp
from flax import linen as nn

import qp


class ActorCriticNetwork(nn.Module):
    action_space: int
    time_horizon: float
    nodes: int
    num_states: int
    mass_cart: float
    mass_pole: float
    length: float
    gravity: float

    def setup(self):
        dtype = jnp.float32
        features = 64
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
        # Setup QP:
        equaility_functions, inequality_functions, objective_functions, linearized_functions = (
            qp.qp_preprocess(
                time_horizon=self.time_horizon,
                nodes=self.nodes,
                num_states=self.num_states,
                mass_cart=self.mass_cart,
                mass_pole=self.mass_pole,
                length=self.length,
                gravity=self.gravity,
            )
        )
        # Isolate Function:
        self.osqp_layer = lambda x, y: qp.qp_layer(
            x,
            y,
            equaility_functions,
            inequality_functions,
            objective_functions,
            linearized_functions,
            self.nodes,
            self.num_states,
        )

    # Small Network:
    def model(self, x):
        range_limit = 0.1

        # QP Layer Inputs:
        initial_condition = x[:4]
        previous_trajectory = jnp.reshape(x[4:], (self.num_states, -1)).T

        # Shared Layers: QP
        state_trajectory, objective_value, status = self.osqp_layer(
            initial_condition,
            previous_trajectory,
        )

        # Policy Layer: Nonlinear Function of Acceleration
        y = self.dense_1(state_trajectory)
        y = nn.tanh(y)
        y = self.dense_2(y)
        y = nn.tanh(y)
        # Pipeline that decides std should have more information of the states
        z = self.dense_3(state_trajectory)
        z = nn.tanh(z)
        z = self.dense_4(z)
        z = nn.tanh(z)

        # Value Layer: Nonlinear Function of Objective Value
        w = self.dense_5(state_trajectory)
        w = nn.tanh(w)
        w = self.dense_6(w)
        w = nn.tanh(w)

        # Output Layer:
        mean = self.mean_layer(y)
        mean = range_limit * nn.tanh(mean)
        std = self.std_layer(z)
        std = nn.softplus(std)
        values = self.value_layer(w)
        return mean, std, values, state_trajectory, objective_value, status

    def __call__(self, x):
        mean, std, values, state_trajectory, objective_value, status = self.model(x)
        return mean, std, values, state_trajectory, objective_value, status


class ActorCriticNetworkVmap(nn.Module):
    action_space: int
    time_horizon: float
    nodes: int
    num_states: int
    mass_cart: float
    mass_pole: float
    length: float
    gravity: float

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
            num_states=self.num_states,
            mass_cart=self.mass_cart,
            mass_pole=self.mass_pole,
            length=self.length,
            gravity=self.gravity,
        )

    def __call__(self, x):
        mean, std, values, state_trajectory, objective_value, status = self.model(x)
        return mean, std, values, state_trajectory, objective_value, status
