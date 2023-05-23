from functools import partial
from typing import Callable
from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP


@partial(jax.jit, static_argnames=['num_states', 'dt'])
def equality_constraints(
    q: jax.typing.ArrayLike,
    initial_conditions: jax.typing.ArrayLike,
    a: jax.typing.ArrayLike,
    f_a: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
    df_dq: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
    num_states: int,
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

    # Function Approximation:
    def func_approximation(
        x: jax.typing.ArrayLike,
        u: jax.typing.ArrayLike,
        a: jax.typing.ArrayLike,
        f_a: jax.typing.ArrayLike,
        df_dq: jax.typing.ArrayLike,
    ) -> jnp.ndarray:
        state = df_dq[:, 0] * x[0][:] + df_dq[:, 1] * x[1][:]
        control = df_dq[:, -1] * u[:]
        const = f_a[:] - df_dq[:, 0] * a[:, 0] - df_dq[:, 1] * a[:, 1] - df_dq[:, -1] * a[:, -1]
        f = state + control + const
        return f

    # Sort State Vector:
    q = q.reshape((num_states, -1))

    # State Nodes:
    x = q[0, :]
    dx = q[1, :]
    th = q[2, :]
    dth = q[3, :]
    u = q[4, :]

    # Linearization Terms:
    f_a_ddx, f_a_ddth = f_a
    df_dq_ddx, df_dq_ddth = df_dq

    initial_condition = jnp.asarray([
        x[0] - initial_conditions[0],
        dx[0] - initial_conditions[1],
        th[0] - initial_conditions[2],
        dth[0] - initial_conditions[3],
    ], dtype=jnp.float32)

    # 2. Collocation Constraints:
    ddx = func_approximation(
        x=jnp.vstack([th, dth]),
        u=u,
        a=a[:, 2:],
        f_a=f_a_ddx,
        df_dq=df_dq_ddx[:, 2:],
    )
    ddth = func_approximation(
        x=jnp.vstack([th, dth]),
        u=u,
        a=a[:, 2:],
        f_a=f_a_ddth,
        df_dq=df_dq_ddth[:, 2:],
    )

    x_defect = collocation_constraint(jnp.vstack([x, dx]), dt)
    dx_defect = collocation_constraint(jnp.vstack([dx, ddx]), dt)
    th_defect = collocation_constraint(jnp.vstack([th, dth]), dt)
    dth_defect = collocation_constraint(jnp.vstack([dth, ddth]), dt)

    equality_constraint = jnp.concatenate(
        [
            initial_condition,
            x_defect,
            dx_defect,
            th_defect,
            dth_defect,
        ]
    )

    return equality_constraint


@partial(jax.jit, static_argnames=['num_states'])
def inequality_constraints(
    q: jax.Array,
    num_states: int,
) -> jnp.ndarray:
    # Sort State Vector:
    q = q.reshape((num_states, -1))

    # State Nodes:
    x = q[0, :]
    dx = q[1, :]
    th = q[2, :]
    dth = q[3, :]
    u = q[4, :]

    # State Limits:
    position_limit = 5.0
    velocity_limit = 10.0
    force_limit = 10.0
    inequality_constraints = jnp.vstack(
        [
            [-x - position_limit],
            [-dx - velocity_limit],
            [-u - force_limit],
        ],
    ).flatten()

    return inequality_constraints


@partial(jax.jit, static_argnames=['num_states'])
def objective_function(
    q: jax.typing.ArrayLike,
    target_position: jax.typing.ArrayLike,
    num_states: int,
) -> jnp.ndarray:
    """
    Objective Function:
        1. Position Target
        2. Control Effort Objective
    """

    # Sort State Vector:
    q = q.reshape((num_states, -1))

    # State Nodes:
    x = q[0, :]
    dx = q[1, :]
    th = q[2, :]
    dth = q[3, :]
    u = q[4, :]

    # Objective Function:
    target_objective = jnp.sum((th - target_position) ** 2, axis=0)

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
    num_states: int,
    mass_cart: float,
    mass_pole: float,
    length: float,
    gravity: float,
) -> Callable:
    # Print Statement:
    print('Running Preprocess...')

    # Optimization Parameters:
    dt = time_horizon / (nodes - 1)

    # Get Dynamics:
    f_ddx, f_ddth = get_dynamics(
        mass_cart=mass_cart,
        mass_pole=mass_pole,
        length=length,
        gravity=gravity,
    )

    # Vmapped Linearized Dynamics:
    linearized_functions = jax.vmap(
        lambda x: linearize_dynamics(
            q=x,
            dynamics_eq=(f_ddx, f_ddth),
            num_vars=num_states,
        ),
        in_axes=0,
    )

    # Isolate Functions to Lambda Functions:
    equality_func = lambda x, ic, a, f_a, df_dq: equality_constraints(
        q=x,
        initial_conditions=ic,
        a=a,
        f_a=f_a,
        df_dq=df_dq,
        num_states=num_states,
        dt=dt,
    )

    inequality_func = lambda x: inequality_constraints(
        q=x,
        num_states=num_states,
    )

    objective_func = lambda x, tp: objective_function(
        q=x,
        target_position=tp,
        num_states=num_states,
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

    return equaility_functions, inequality_functions, objective_functions, linearized_functions


@partial(jax.jit, static_argnames=['equaility_functions', 'inequality_functions', 'objective_functions', 'linearized_functions', 'nodes', 'num_states'])
def qp_layer(
    initial_conditions: jax.typing.ArrayLike,
    target_position: jax.typing.ArrayLike,
    previous_trajectory: jax.typing.ArrayLike,
    equaility_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    linearized_functions: Callable,
    nodes: int,
    num_states: int,
) -> jnp.ndarray:
    # Print Statement:
    print('Running QP Layer...')

    # Unpack Functions:
    b_eq_fn, A_eq_fn = equaility_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Optimization Variables:
    setpoint = jnp.zeros(
        (num_states * nodes,),
        dtype=jnp.float32,
    )

    # Get Linearizations:
    previous_trajectory = jnp.reshape(previous_trajectory, (-1, num_states))
    linear_ddx, linear_ddth = linearized_functions(previous_trajectory)
    f_a_ddx, df_dq_ddx = linear_ddx
    f_a_ddth, df_dq_ddth = linear_ddth

    # Generate QP Matrices:
    A_eq = A_eq_fn(
        setpoint,
        initial_conditions,
        previous_trajectory,
        (f_a_ddx, f_a_ddth),
        (df_dq_ddx, df_dq_ddth),
    )
    b_eq = -b_eq_fn(
        setpoint,
        initial_conditions,
        previous_trajectory,
        (f_a_ddx, f_a_ddth),
        (df_dq_ddx, df_dq_ddth),
    )

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

    state_trajectory = jnp.reshape(sol.primal[0], (num_states, -1))
    objective_value = objective_fn(sol.primal[0], target_position)

    return state_trajectory, objective_value, state


@partial(jax.jit, static_argnames=['dynamics_eq', 'num_vars'])
def linearize_dynamics(
    q: jax.typing.ArrayLike,
    dynamics_eq: tuple[Callable, Callable],
    num_vars: int,
) -> jnp.ndarray:
    """
        q: Linearization point -> state order of [x, dx, th, dth, u]
        dynamics_eq: isolated function handle for acceleration equations -> (ddx, ddth)
    """
    # Unpack Tuple:
    f_ddx, f_ddth = dynamics_eq

    # Tangent Input: State Mask
    in_tangents = jnp.eye(num_vars)

    # Linearize ddx equation:
    jvp_ddx = partial(jax.jvp, f_ddx, (q,))
    f_a_ddx, del_ddx_del_q = jax.vmap(
        jvp_ddx,
        out_axes=(None, 0)
    )((in_tangents,))

    # Linearize ddth equation:
    jvp_ddth = partial(jax.jvp, f_ddth, (q,))
    f_a_ddth, del_ddth_del_q = jax.vmap(
        jvp_ddth,
        out_axes=(None, 0)
    )((in_tangents,))

    linear_ddx = (f_a_ddx, del_ddx_del_q)
    linear_ddth = (f_a_ddth, del_ddth_del_q)

    return linear_ddx, linear_ddth


def get_dynamics(
    mass_cart: float,
    mass_pole: float,
    length: float,
    gravity: float,
) -> tuple[Callable, Callable]:
    """
    Creates isolated function handles for cart pole acceleration equations.
    """
    def ddx(
            q: jax.typing.ArrayLike,
            mass_cart: float,
            mass_pole: float,
            length: float,
            gravity: float,
    ) -> jnp.ndarray:
        # States corresponding to the point of linearization:
        x = q[0]
        dx = q[1]
        th = q[2]
        dth = q[3]
        u = q[4]

        # Equation of Motion for ddx:
        a = u / (mass_cart + mass_pole * jnp.sin(th) ** 2)
        b = mass_pole * length * dth ** 2 * jnp.sin(th) / (mass_cart + mass_pole * jnp.sin(th) ** 2)
        c = mass_pole * gravity * jnp.sin(th) * jnp.cos(th) / (mass_cart + mass_pole * jnp.sin(th) ** 2)
        return a + b + c

    def ddth(
            q: jax.typing.ArrayLike,
            mass_cart: float,
            mass_pole: float,
            length: float,
            gravity: float,
    ) -> jnp.ndarray:
        # States corresponding to the point of linearization:
        x = q[0]
        dx = q[1]
        th = q[2]
        dth = q[3]
        u = q[4]

        # Equation of Motion for ddth:
        a = -u * jnp.cos(th) / (mass_cart * length + mass_pole * length * jnp.sin(th) ** 2)
        b = -mass_pole * length * dth ** 2 * jnp.sin(th) * jnp.cos(th) / (mass_cart * length + mass_pole * length * jnp.sin(th) ** 2)
        c = -(mass_cart + mass_pole) * gravity * jnp.sin(th) / (mass_cart * length + mass_pole * length * jnp.sin(th) ** 2)
        return a + b + c

    # Isolated Functions:
    f_ddx = lambda x: ddx(
        x,
        mass_cart=mass_cart,
        mass_pole=mass_pole,
        length=length,
        gravity=gravity,
    )
    f_ddth = lambda x: ddth(
        x,
        mass_cart=mass_cart,
        mass_pole=mass_pole,
        length=length,
        gravity=gravity,
    )

    return f_ddx, f_ddth


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
    num_vars = 5
    linearization_point = jnp.ones((5, num_vars), dtype=jnp.float32)

    # Get Dynamics:
    f_ddx, f_ddth = get_dynamics(
        mass_cart=1.,
        mass_pole=1.,
        length=1.,
        gravity=9.81,
    )

    # Linearize Dynamics:
    linearized_functions = jax.vmap(
        lambda x: linearize_dynamics(
            q=x,
            dynamics_eq=(f_ddx, f_ddth),
            num_vars=num_vars,
        ),
        in_axes=0,
    )

    linear_ddx, linear_ddth = linearized_functions(linearization_point)
    print(linear_ddx)
    print(linear_ddth)


if __name__ == '__main__':
    app.run(main)
