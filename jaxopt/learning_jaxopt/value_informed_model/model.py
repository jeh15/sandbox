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
        # Used in Value Function Averaging
        # self.dense_7 = nn.Dense(
        #     features=features,
        #     name='dense_7',
        #     dtype=dtype,
        # )
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

    # Value Function Averaging: Not Good
    # Idea: Let network model value function from ic, pos, vel, acc and average it with function derived from obj val
    # def model(self, x):
    #     range = 2.0

    #     # QP Layer Inputs:
    #     initial_conditions = x
    #     target_position = jnp.array([1.0])

    #     # Shared Layers: QP
    #     pos, vel, acc, obj_val, status = self.osqp_layer(
    #         initial_conditions, target_position
    #     )
    #     trajectory = jnp.vstack([pos, vel, acc])
    #     # Change to Maximization Problem:
    #     obj_val = jnp.expand_dims(-obj_val, axis=-1)

    #     # Policy Layer:
    #     policy_input = jnp.concatenate([initial_conditions, pos, vel, acc], axis=0)
    #     y = self.dense_1(policy_input)
    #     y = nn.tanh(y)
    #     y = self.dense_2(y)
    #     y = nn.tanh(y)
    #     z = self.dense_3(policy_input)
    #     z = nn.tanh(z)
    #     z = self.dense_4(z)
    #     z = nn.tanh(z)

    #     # Value Layer: Nonlinear Relationship
    #     nn_value = self.dense_5(policy_input)
    #     nn_value = nn.tanh(nn_value)
    #     nn_value = self.dense_6(nn_value)
    #     qp_value = self.dense_7(obj_val)
    #     average_value = jnp.mean(jnp.vstack([nn_value, qp_value]), axis=0)

    #     # # Value Layer: Linear Relationship
    #     # nn_value = self.dense_5(policy_input)
    #     # qp_value = self.dense_6(obj_val)
    #     # average_value = jnp.mean(jnp.vstack([nn_value, qp_value]), axis=0)

    #     # Output Layer:
    #     mean = self.mean_layer(y)
    #     mean = range * nn.tanh(mean)
    #     std = self.std_layer(z)
    #     std = nn.softplus(std)
    #     values = self.value_layer(average_value)
    #     return mean, std, values, trajectory, obj_val, status

    # # Full Information Propogation Network:
    # def model(self, x):
    #     range = 2.0

    #     # QP Layer Inputs:
    #     initial_conditions = x
    #     target_position = jnp.array([1.0])

    #     # Shared Layers: QP
    #     pos, vel, acc, obj_val, status = self.osqp_layer(
    #         initial_conditions, target_position
    #     )
    #     trajectory = jnp.vstack([pos, vel, acc])
    #     # Change to Maximization Problem:
    #     obj_val = jnp.expand_dims(-obj_val, axis=-1)

    #     # Policy Layer:
    #     policy_input = jnp.concatenate([pos, vel, acc], axis=0)
    #     y = self.dense_1(policy_input)
    #     y = nn.tanh(y)
    #     y = self.dense_2(y)
    #     y = nn.tanh(y)
    #     z = self.dense_3(policy_input)
    #     z = nn.tanh(z)
    #     z = self.dense_4(z)
    #     z = nn.tanh(z)

    #     # Value Layer:
    #     value_input = jnp.concatenate([pos, vel, acc, obj_val], axis=0)
    #     w = self.dense_5(value_input)
    #     w = nn.tanh(w)
    #     w = self.dense_6(w)

    #     # Output Layer:
    #     mean = self.mean_layer(y)
    #     mean = range * nn.tanh(mean)
    #     std = self.std_layer(z)
    #     std = nn.softplus(std)
    #     values = self.value_layer(w)
    #     return mean, std, values, trajectory, obj_val, status

    # Small Network: Nonlinear Policy + Value Layer -- More Information Propogation
    def model(self, x):
        range = 2.0

        # QP Layer Inputs:
        initial_conditions = x
        target_position = jnp.array([1.0])

        # Shared Layers: QP
        pos, vel, acc, obj_val, status = self.osqp_layer(
            initial_conditions, target_position
        )
        trajectory = jnp.vstack([pos, vel, acc])
        # Change to Maximization Problem:
        obj_val = jnp.expand_dims(-obj_val, axis=-1)

        # Policy Layer: Nonlinear Function of Acceleration
        # The pipeline that decides the mean should only be function of acceleration
        y = self.dense_1(acc)
        y = nn.tanh(y)
        y = self.dense_2(y)
        y = nn.tanh(y)
        # Pipeline that decides std should have more information of the states
        predicted_states = jnp.concatenate([pos, vel, acc], axis=0)
        z = self.dense_3(predicted_states)
        z = nn.tanh(z)
        z = self.dense_4(z)
        z = nn.tanh(z)

        # Value Layer: Nonlinear Function of Objective Value
        # Value function needs to be a function of states and objective value
        value_layer_input = jnp.concatenate([predicted_states, obj_val], axis=0)
        w = self.dense_5(value_layer_input)
        w = nn.tanh(w)
        w = self.dense_6(w)
        w = nn.tanh(w)

        # Output Layer:
        mean = self.mean_layer(y)
        mean = range * nn.tanh(mean)
        std = self.std_layer(z)
        std = nn.softplus(std)
        values = self.value_layer(w)
        return mean, std, values, trajectory, obj_val, status
    
    # # Small Network: Nonlinear Policy + Value Layer -> ~-29
    # def model(self, x):
    #     range = 2.0

    #     # QP Layer Inputs:
    #     initial_conditions = x
    #     target_position = jnp.array([1.0])

    #     # Shared Layers: QP
    #     pos, vel, acc, obj_val, status = self.osqp_layer(
    #         initial_conditions, target_position
    #     )
    #     trajectory = jnp.vstack([pos, vel, acc])
    #     # Change to Maximization Problem:
    #     obj_val = jnp.expand_dims(-obj_val, axis=-1)

    #     # Policy Layer: Nonlinear Function of Acceleration
    #     y = self.dense_1(acc)
    #     y = nn.tanh(y)
    #     y = self.dense_2(y)
    #     y = nn.tanh(y)
    #     z = self.dense_3(initial_conditions)
    #     z = nn.tanh(z)
    #     z = self.dense_4(z)
    #     z = nn.tanh(z)

    #     # Value Layer: Nonlinear Function of Objective Value
    #     w = self.dense_5(obj_val)
    #     w = nn.tanh(w)
    #     w = self.dense_6(w)
    #     w = nn.tanh(w)

    #     # Output Layer:
    #     mean = self.mean_layer(y)
    #     mean = range * nn.tanh(mean)
    #     std = self.std_layer(z)
    #     std = nn.softplus(std)
    #     values = self.value_layer(w)
    #     return mean, std, values, trajectory, obj_val, status


    # # Small Network: Nonlinear Policy Layer
    # def model(self, x):
    #     range = 2.0

    #     # QP Layer Inputs:
    #     initial_conditions = x
    #     target_position = jnp.array([1.0])

    #     # Shared Layers: QP
    #     pos, vel, acc, obj_val, status = self.osqp_layer(
    #         initial_conditions, target_position
    #     )
    #     trajectory = jnp.vstack([pos, vel, acc])
    #     # Change to Maximization Problem:
    #     obj_val = jnp.expand_dims(-obj_val, axis=-1)

    #     # Policy Layer: Nonlinear Function of Acceleration
    #     y = self.dense_1(acc)
    #     y = nn.tanh(y)
    #     y = self.dense_2(y)
    #     y = nn.tanh(y)
    #     z = self.dense_3(initial_conditions)
    #     z = nn.tanh(z)
    #     z = self.dense_4(z)
    #     z = nn.tanh(z)

    #     # Value Layer: Linear Function of Objective Value
    #     w = self.dense_5(obj_val)
    #     w = self.dense_6(w)

    #     # Output Layer:
    #     mean = self.mean_layer(y)
    #     mean = range * nn.tanh(mean)
    #     std = self.std_layer(z)
    #     std = nn.softplus(std)
    #     values = self.value_layer(w)
    #     return mean, std, values, trajectory, obj_val, status

    # # Small Network: Linear Policy Layer -> At worst performs same as nonlinear at best ~-17
    # # Performs the best/learns fastest out of ALL QP Layers...
    # # (NOTE: Oops forgot to add activation functions... but it performed well with only linear approximations?)
    # def model(self, x):
    #     range = 2.0

    #     # QP Layer Inputs:
    #     initial_conditions = x
    #     target_position = jnp.array([1.0])

    #     # Shared Layers: QP
    #     pos, vel, acc, obj_val, status = self.osqp_layer(
    #         initial_conditions, target_position
    #     )
    #     trajectory = jnp.vstack([pos, vel, acc])
    #     # Change to Maximization Problem:
    #     obj_val = jnp.expand_dims(-obj_val, axis=-1)

    #     # Policy Layer: Linear Function of Acceleration -- This Performed better than the nonlinear function.
    #     y = self.dense_1(acc)
    #     y = self.dense_2(y)
    #     z = self.dense_3(acc)
    #     z = self.dense_4(z)

    #     # Value Layer: Linear Function of Objective Value
    #     w = self.dense_5(obj_val)
    #     w = self.dense_6(w)

    #     # Output Layer:
    #     mean = self.mean_layer(y)
    #     mean = range * nn.tanh(mean)
    #     std = self.std_layer(z)
    #     std = nn.softplus(std)
    #     values = self.value_layer(w)
    #     return mean, std, values, trajectory, obj_val, status

    # # Linear QP Network: Does not perform well...
    # def model(self, x):
    #     range = 2.0
    #     # QP Layer Inputs:
    #     initial_conditions = x
    #     target_position = jnp.array([1.0])
    #     # Shared Layers: QP
    #     pos, vel, acc, obj_val, status = self.osqp_layer(
    #         initial_conditions, target_position
    #     )
    #     trajectory = jnp.vstack([pos, vel, acc])
    #     # Change to Maximization Problem:
    #     obj_val = jnp.expand_dims(-obj_val, axis=-1)
    #     # Output Layer: Straight Forward Mappings
    #     mean = self.mean_layer(acc)
    #     mean = range * nn.tanh(mean)
    #     std = self.std_layer(acc)
    #     std = nn.softplus(std)
    #     values = self.value_layer(obj_val)
    #     return mean, std, values, trajectory, obj_val, status

    def __call__(self, x):
        mean, std, values, trajectory, obj_val, status = self.model(x)
        return mean, std, values, trajectory, obj_val, status


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
        mean, std, values, trajectory, obj_val, status = self.model(x)
        return mean, std, values, trajectory, obj_val, status
