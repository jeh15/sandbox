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
    position_limit = 5.0
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

    objective_function = jnp.sum(
        jnp.hstack(
            [
                target_objective,
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
    num_vars = 3
    setpoint = jnp.zeros(
        (num_vars * nodes,),
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

    qp = BoxOSQP(
        momentum=1.6,
        primal_infeasible_tol=1e-3,
        dual_infeasible_tol=1e-3,
        rho_start=1e-2,
        maxiter=400,
        tol=1e-2,
        verbose=0,
        jit=True,
    )

    # Solve QP:
    sol, state = qp.run(
        params_obj=(H, f),
        params_eq=A,
        params_ineq=(lb, ub),
    )

    pos = sol.primal[0][:nodes]
    vel = sol.primal[0][nodes:-nodes]
    acc = sol.primal[0][-nodes:]

    objective_value = objective_fn(sol.primal[0], target_position)

    return pos, vel, acc, objective_value, state


# This needs to be reworked:
def dynamics(q: jax.typing.ArrayLike) -> jnp.ndarray:
    def ddx(q, m1, m2, l, g):
        # States corresponding to the point of linearization:
        x = q[0]
        dx = q[1]
        th = q[2]
        dth = q[3]
        u = q[4]

        # Equation of Motion for ddx:
        a = u / (m1 + m2 * jnp.sin(th) ** 2)
        b = m2 * l * dth ** 2 * jnp.sin(th) / (m1 + m2 * jnp.sin(th) ** 2)
        c = m2 * g * jnp.sin(th) * jnp.cos(th) / (m1 + m2 * jnp.sin(th) ** 2)
        return a + b + c

    def ddth(q: jax.typing.ArrayLike, m1: float, m2: float, l: float, g: float) -> jnp.ndarray:
        # States corresponding to the point of linearization:
        x = q[0]
        dx = q[1]
        th = q[2]
        dth = q[3]
        u = q[4]

        # Equation of Motion for ddth:
        a = -u * jnp.cos(th) / (m1 * l + m2 * l * jnp.sin(th) ** 2)
        b = -m2 * l * dth ** 2 * jnp.sin(th) * jnp.cos(th) / (m1 * l + m2 * l * jnp.sin(th) ** 2)
        c = -(m1 + m2) * g * jnp.sin(th) / (m1 * l + m2 * l * jnp.sin(th) ** 2)
        return a + b + c

    # Linearize Dynamics:
    m1 = 1.0
    m2 = 1.0
    l = 1.0
    g = 9.81
    f_x = lambda x: ddx(x, m1, m2, l, g)
    f_th = lambda x: ddth(x, m1, m2, l, g)
    y, f_jvp = jax.linearize(f_x, q)
    return y, f_jvp


def main(argv=None):
    def f(x): return x[0] ** 2 + x[1] ** 2

    jacobian = jax.jacfwd(f)(jnp.array([1., 1.], dtype=jnp.float32))
    manual_tangets = jacobian @ jnp.array([[1., 0.], [0., 1.]])
    primals, tangets = jax.jvp(f, ([1., 1.],), ([1., 1.],))
    out, fjvp = jax.linearize(f, jnp.array([1., 1.]))

    print("Manual Calculation:")
    print(jacobian)
    print(manual_tangets)

    print("JVP Calculation:")
    print(primals)
    print(tangets)

    print("Linearization Calculation:")
    print(out)
    print(fjvp(jnp.array([1., 0.])))
    print(fjvp(jnp.array([0., 1.])))

    """
    Linearization:
    eq: p(x) = f(a) + f'(a)(x - a)

    Jax Terminology:
    primal_in: Linearization point (a)
    tangent_in: The vector that gets multiplied by the Jacobian at point (a)
    primal_out: f(a)
    tangent_out = jac(f)(a) @ tangent_in

    What does this give us?
    primal_out provides f(a) from the eq.
    tangent_out can provide f'(a) if tangent_in is used as a masking vector.

    Ex:
    (Note: to evaluate multiple derivatives, the input needs to be a vector)

    Linearize the following function at point (1, 1):
    f(x) = x[0]^2 + x[1]^2
    a = (1., 1.)

    Solution:
    p(x) = f(a) + df/dx1(a)(x1 - a1) + df/dx2(a)(x2 - a2)

    Calculations:
    f(a) = 2.

    jac(f) = [2x[0], 2x[1]]
    jac(f)(a) = [2., 2.]

    df/dx1(a) = jac(f)(a)[0] -> jac(f)(a) @ [1., 0.] = 2.
    df/dx2(a) = jac(f)(a)[1] -> jac(f)(a) @ [0., 1.] = 2.

    p(x) = 2. + 2.(x1 - 1.) + 2.(x2 - 1.)

    """

    # Linearize Dynamics:
    linearization_point = jnp.array([1., 1., 1., 1., 1.], dtype=jnp.float32)

    # Dynamics:
    def ddx(q, m1, m2, l, g):
        # States corresponding to the point of linearization:
        x = q[0]
        dx = q[1]
        th = q[2]
        dth = q[3]
        u = q[4]

        # Equation of Motion for ddx:
        a = u / (m1 + m2 * jnp.sin(th) ** 2)
        b = m2 * l * dth ** 2 * jnp.sin(th) / (m1 + m2 * jnp.sin(th) ** 2)
        c = m2 * g * jnp.sin(th) * jnp.cos(th) / (m1 + m2 * jnp.sin(th) ** 2)
        return a + b + c

    # Isolate equation:
    m1, m2, l, g = 1.0, 1.0, 1.0, 9.81
    f_ddx = lambda x: ddx(x, m1, m2, l, g)

    f_a, f_jvp = jax.linearize(f_ddx, linearization_point)
    df_x = f_jvp(jnp.array([1., 0., 0., 0., 0.]))
    df_dx = f_jvp(jnp.array([0., 1., 0., 0., 0.]))
    df_th = f_jvp(jnp.array([0., 0., 1., 0., 0.]))
    df_dth = f_jvp(jnp.array([0., 0., 0., 1., 0.]))
    df_u = f_jvp(jnp.array([0., 0., 0., 0., 1.]))

    print("Linearization of ddx:")
    print(f_a)
    print(df_x)
    print(df_dx)
    print(df_th)
    print(df_dth)
    print(df_u)


if __name__ == '__main__':
    app.run(main)
