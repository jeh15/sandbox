from dataclasses import dataclass

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev

import numpy as np

import pdb

key = random.PRNGKey(0)


@dataclass
class params:
    nodes: int
    time_horizon: float
    friction: float
    mass: float
    W: np.ndarray
    position_target: np.ndarray

    def __post_init__(self):
        self.dt = self.time_horizon / (self.nodes - 1)


def hessian(f):
    return jacfwd(jacrev(f))


def objective_function(x, params):
    # Extract Matrices for easier manipulation:
    state_position = x[:2, :]
    state_control_input = x[-2:, :]

    """
    Objective Function:
    """

    state_error = state_position - params.position_target
    J_minimize_error = params.W[0] * (state_error ** 2)
    J_minimize_effort = params.W[1] * (state_control_input ** 2)
    J = jnp.sum(jnp.sum(J_minimize_error, axis=0) + jnp.sum(J_minimize_effort, axis=0))
    return J


def main():
    # Inital Params:
    opt_params = params(21, 1.0, 0.1, 1.0, jnp.array([0.1, 0.01], dtype=float), jnp.ones((2, 1), dtype=float))
    dummy_input = jnp.zeros((6, 21), dtype=float)
    objective_function(dummy_input, opt_params)

    # Isolate Function:
    func = lambda x: objective_function(x, opt_params)
    set_point = jnp.zeros((6, 21), dtype=float)
    H = hessian(func)(set_point)
    H = H.reshape(H.shape[0] * H.shape[1], H.shape[2] * H.shape[3])
    f = jacfwd(func)(set_point)
    f = f.reshape(f.shape[0] * f.shape[1], -1)
    print(f"H matrix: {H.shape}")
    print(f"f matrix: {f.shape}")


if __name__ == "__main__":
    main()
