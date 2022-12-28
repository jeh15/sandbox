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
    ic: np.ndarray

    def __post_init__(self):
        self.dt = self.time_horizon / (self.nodes - 1)


def equality_constraints(x, params):
    # Extract Matrices for easier manipulation:
    state_x = x[0, :]
    state_y = x[1, :]
    state_dx = x[2, :]
    state_dy = x[3, :]
    state_ux = x[4, :]
    state_uy = x[5, :]

    """
    Equality Constraints:

    1. Initial Conditions

    2. Collocation Constraints

    """

    # 1. Initial Condition Constraints:
    initial_condition = jnp.array([
        state_x[0] - params.ic[0], state_y[0] - params.ic[1],
        state_dx[0] - params.ic[2], state_dy[0] - params.ic[3],
        state_ux[0] - params.ic[4], state_uy[0] - params.ic[5],
    ], dtype=float)

    # 2. Collocation Constraints:
    def collocation_func(z):
        return z[0][1:] - z[0][:-1] - z[1][:-1] * params.dt

    state_ddx = (state_ux - params.friction * state_dx) / params.mass
    state_ddy = (state_uy - params.friction * state_dy) / params.mass
    state_x_defect = collocation_func([state_x, state_dx])
    state_y_defect = collocation_func([state_y, state_dy])
    state_dx_defect = collocation_func([state_dx, state_ddx])
    state_dy_defect = collocation_func([state_dy, state_ddy])

    # Combine Constraints:
    equality_constraint = jnp.concatenate(
        (
            initial_condition,
            state_x_defect,
            state_y_defect,
            state_dx_defect,
            state_dy_defect,
        )
    )

    return equality_constraint


def main():
    # Inital Params:
    opt_params = params(21, 1.0, 0.1, 1.0, jnp.ones((6,), dtype=float))

    # Isolate Function:
    f = lambda x: equality_constraints(x, opt_params)
    set_point = jnp.zeros((6, 21), dtype=float)
    A = jacfwd(f)(set_point)
    A = A.reshape(A.shape[0], -1)
    b = equality_constraints(set_point, opt_params)
    b = b.reshape(b.shape[0], 1)
    print(f"A matrix: {A.shape}")
    print(f"b matrix: {b.shape}")


if __name__ == "__main__":
    main()
