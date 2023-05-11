from functools import partial
from typing import Callable

from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP


@partial(jax.jit, static_argnames=['dt'])
def equality_constraints(
    q: jax.typing.ArrayLike,
    initial_conditions: jax.typing.ArrayLike,
    dt: float,
) -> jnp.ndarray:
    """
    Equality Constraints:
        1. Initial Condition
        2. Collocation Constraint
    """

    # Euler Collocation:
    def collocation_constraint(
        x: jax.typing.ArrayLike,
        dt: float,
    ) -> jnp.ndarray:
        collocation = x[0][1:] - x[0][:-1] - x[1][:-1] * dt
        return collocation

    # Sort State Vector:
    q = q.reshape((3, -1))

    # State Nodes:
    x = q[0, :]
    dx = q[1, :]
    ux = q[2, :]

    initial_condition = jnp.asarray([
        x[0] - initial_conditions[0],
        dx[0] - initial_conditions[1],
    ], dtype=float)

    # 2. Collocation Constraints:
    # Model Parameters: (Match Brax model future make these learnable)
    friction = 0.05
    mass = 1.0

    ddx = (ux - friction * dx) / mass
    x_defect = collocation_constraint([x, dx], dt)
    dx_defect = collocation_constraint([dx, ddx], dt)

    equality_constraint = jnp.concatenate(
        [
            initial_condition,
            x_defect,
            dx_defect,
        ]
    )

    return equality_constraint


@jax.jit
def inequality_constraints(
    q: jax.Array,
) -> jnp.ndarray:
    # Sort State Vector:
    q = q.reshape((3, -1))

    # State Nodes:
    x = q[0, :]
    dx = q[1, :]
    ux = q[2, :]

    # State Limits:
    position_limit = 10.0
    velocity_limit = 10.0
    force_limit = 10.0
    inequality_constraints = jnp.vstack(
        [
            [-x - position_limit],
            [-dx - velocity_limit],
            [-ux - force_limit],
        ],
    ).flatten()

    return inequality_constraints


@jax.jit
def objective_function(
    q: jax.typing.ArrayLike,
    target_position: jax.typing.ArrayLike,
) -> jnp.ndarray:
    """
    Objective Function:
        1. Position Target
        2. Control Effort Objective
    """

    # Sort State Vector:
    q = q.reshape((3, -1))

    # State Nodes:
    x = q[0, :]
    dx = q[1, :]
    ux = q[2, :]

    # Objective Function:
    target_objective = jnp.sum((x - target_position) ** 2, axis=0)
    minimize_control = jnp.sum(ux ** 2, axis=0)

    objective_function = jnp.sum(
        jnp.hstack(
            [
                target_objective,
                minimize_control,
            ],
        ),
        axis=0,
    )

    return objective_function


def qp_preprocess(
    time_horizon: float,
    nodes: int,
) -> Callable:
    # Print Statement:
    print('Running Preprocess...')

    # Optimization Parameters:
    dt = time_horizon / (nodes - 1)

    # Isolate Functions to Lambda Functions:
    equality_func = lambda x, ic: equality_constraints(
        q=x,
        initial_conditions=ic,
        dt=dt,
    )

    inequality_func = lambda x: inequality_constraints(
        q=x,
    )

    objective_func = lambda x, tp: objective_function(
        q=x,
        target_position=tp,
    )

    # Function generates A matrix for equality constraints:
    A_eq_fn = jax.jit(jax.jacfwd(equality_func))

    # Function generates A matrix for inequality constraints:
    A_ineq_fn = jax.jit(jax.jacfwd(inequality_func))

    # Functions generate H and f matrices for objective function:
    H_fn = jax.jit(jax.jacfwd(jax.jacrev(objective_func)))
    f_fn = jax.jit(jax.jacfwd(objective_func))

    # Package Return Functions:
    equaility_functions = (equality_func, A_eq_fn)
    inequality_functions = (inequality_func, A_ineq_fn)
    objective_functions = (objective_func, H_fn, f_fn)

    return equaility_functions, inequality_functions, objective_functions


@partial(jax.jit, static_argnames=['equaility_functions', 'inequality_functions', 'objective_functions', 'nodes'])
def qp_layer(
    initial_conditions: jax.typing.ArrayLike,
    target_position: jax.typing.ArrayLike,
    equaility_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    nodes: int,
) -> jnp.ndarray:
    # Print Statement:
    print('Running QP Layer...')

    # Unpack Functions:
    b_eq_fn, A_eq_fn = equaility_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Optimization Variables:
    setpoint = jnp.zeros(
        (3 * nodes,),
        dtype=jnp.float32,
    )

    # Generate QP Matrices:
    A_eq = A_eq_fn(setpoint, initial_conditions)
    b_eq = -b_eq_fn(setpoint, initial_conditions)

    A_ineq = A_ineq_fn(setpoint)
    b_ineq_lb = b_ineq_fn(setpoint)
    b_ineq_ub = -b_ineq_fn(setpoint)

    H = H_fn(setpoint, target_position)
    f = f_fn(setpoint, target_position)

    A = jnp.vstack(
        [A_eq, A_ineq],
    )
    lb = jnp.concatenate(
        [b_eq, b_ineq_lb],
        axis=0,
    )
    ub = jnp.concatenate(
        [b_eq, b_ineq_ub],
        axis=0,
    )

    # # class attributes (ignored by @dataclass)
    # UNSOLVED          = 0  # stopping criterion not reached yet.
    # SOLVED            = 1  # feasible solution found with satisfying precision.
    # DUAL_INFEASIBLE   = 2  # infeasible dual (infeasible primal or unbounded primal).
    # PRIMAL_INFEASIBLE = 3  # infeasible primal

    # Create QP:
    qp = BoxOSQP(
        check_primal_dual_infeasability=False,
        eq_qp_solve='cg+jacobi',
        primal_infeasible_tol=1e-3,
        dual_infeasible_tol=1e-3,
        rho_start=1e-2,
        maxiter=2000,
        tol=1e-3,
        verbose=0,
        jit=True,
    )

    # # Solve QP:
    # sol, state = qp.run(
    #     params_obj=(H, f),
    #     params_eq=A,
    #     params_ineq=(lb, ub),
    # )

    # Solve QP:
    sol, _ = qp.run(
        params_obj=(H, f),
        params_eq=A,
        params_ineq=(lb, ub),
    )

    pos = sol.primal[0][:nodes]
    vel = sol.primal[0][nodes:-nodes]
    acc = sol.primal[0][-nodes:]

    return pos, vel, acc
