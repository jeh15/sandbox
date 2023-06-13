from functools import partial
from typing import Callable
from absl import app

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jaxopt import BoxOSQP

# Testing QP:
from typing import Optional
import cartpole
import custom_wrapper
from brax.envs import wrapper
from brax.envs.env import Env
import visualize_cartpole as visualizer
import osqp
from scipy import sparse


# @partial(jax.jit, static_argnames=['num_states', 'dt'])
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
    ], dtype=jnp.float64)

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


# @partial(jax.jit, static_argnames=['num_states'])
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


# @partial(jax.jit, static_argnames=['num_states'])
def objective_function(
    q: jax.typing.ArrayLike,
    a: jax.typing.ArrayLike,
    f_a: tuple[jax.typing.ArrayLike, ...],
    df_dq: tuple[jax.typing.ArrayLike, ...],
    num_states: int,
) -> jnp.ndarray:
    """
    Objective Function:
        1. Linearized cos(x)
    """

    # Function Approximation: TO DO
    # cos(x) = cos(a) - sin(a) * (x - a)
    def func_approximation(
        x: jax.typing.ArrayLike,
        a: jax.typing.ArrayLike,
        f_a: jax.typing.ArrayLike,
        df_dq: jax.typing.ArrayLike,
    ) -> jnp.ndarray:
        state = df_dq[:, 2] * x[2][:]
        const = f_a[:] + df_dq[:, 2] * a[:, 2]
        f = state + const
        return f

    # Sort State Vector and Unpack:
    q = q.reshape((num_states, -1))

    # State Nodes:
    x = q[0, :]
    dx = q[1, :]
    th = q[2, :]
    dth = q[3, :]
    u = q[4, :]

    # Linearization Terms:
    f_a, = f_a
    df_dq, = df_dq

    # Objective Function:
    obj_value = func_approximation(
        x=q,
        a=a,
        f_a=f_a,
        df_dq=df_dq,
    )

    objective_function = jnp.sum(
        jnp.hstack(
            [
                obj_value,
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
    # Optimization Parameters:
    dt = time_horizon / (nodes - 1)

    # Get Dynamics:
    f_ddx, f_ddth, f_obj = get_nonlinear_equations(
        mass_cart=mass_cart,
        mass_pole=mass_pole,
        length=length,
        gravity=gravity,
    )

    # Vmapped Linearized Dynamics:
    linearized_functions = jax.vmap(
        lambda x: linearize_equations(
            q=x,
            dynamics_eq=(f_ddx, f_ddth, f_obj),
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

    objective_func = lambda x, a, f_a, df_dq: objective_function(
        q=x,
        a=a,
        f_a=f_a,
        df_dq=df_dq,
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


# @partial(jax.jit, static_argnames=['equaility_functions', 'inequality_functions', 'objective_functions', 'linearized_functions', 'nodes', 'num_states'])
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
        dtype=jnp.float64,
    )

    # Get Linearizations:
    # previous_trajectory shape -> (nodes, num_states)
    linear_ddx, linear_ddth, linear_obj = linearized_functions(
        previous_trajectory,
    )
    f_a_ddx, df_dq_ddx = linear_ddx
    f_a_ddth, df_dq_ddth = linear_ddth
    f_a_obj, df_dq_obj = linear_obj

    # Generate QP Matrices: START FROM HERE
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

    H = sparse.csc_matrix(
        np.asarray(
            H_fn(
                setpoint,
                previous_trajectory,
                (f_a_obj,),
                (df_dq_obj,),
            ),
        ),
    )
    f = np.asarray(
        f_fn(
            setpoint,
            previous_trajectory,
            (f_a_obj,),
            (df_dq_obj,),
        ),
    )

    A = sparse.csc_matrix(np.asarray(jnp.vstack(
        [A_eq, A_ineq],
    )))
    lb = np.asarray(jnp.concatenate(
        [b_eq, b_ineq_lb],
        axis=0,
    ))
    ub = np.asarray(jnp.concatenate(
        [b_eq, b_ineq_ub],
        axis=0,
    ))

    mp = osqp.OSQP()
    mp.setup(
        P=H,
        q=f,
        A=A,
        l=lb,
        u=ub,
        rho=1e-2,
        max_iter=2000,
        eps_abs=1e-2,
        eps_prim_inf=1e-3,
        eps_dual_inf=1e-3,
        polish=True,
    )
    results = mp.solve()

    state_trajectory = jnp.reshape(results.x, (num_states, -1))
    objective_value = results.info.obj_val
    if results.info.status_val == 1 or results.info.status_val == 2:
        status = 1
    else:
        status = 0

    return state_trajectory, objective_value, status


# @partial(jax.jit, static_argnames=['dynamics_eq', 'num_vars'])
def linearize_equations(
    q: jax.typing.ArrayLike,
    dynamics_eq: tuple[Callable, Callable, Callable],
    num_vars: int,
) -> jnp.ndarray:
    """
        q: Linearization point -> state order of [x, dx, th, dth, u]
        dynamics_eq: isolated function handle for equations -> (ddx, ddth, obj)
    """
    # Unpack Tuple:
    f_ddx, f_ddth, f_obj = dynamics_eq

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

    # Linearize objective function:
    jvp_obj = partial(jax.jvp, f_obj, (q,))
    f_a_obj, del_dobj_del_q = jax.vmap(
        jvp_obj,
        out_axes=(None, 0)
    )((in_tangents,))

    linear_ddx = (f_a_ddx, del_ddx_del_q)
    linear_ddth = (f_a_ddth, del_ddth_del_q)
    linear_obj = (f_a_obj, del_dobj_del_q)

    return linear_ddx, linear_ddth, linear_obj


def get_nonlinear_equations(
    mass_cart: float,
    mass_pole: float,
    length: float,
    gravity: float,
) -> tuple[Callable, Callable, Callable]:
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

    # Trivial but can be used later as a template:
    def obj_func(
        q: jax.typing.ArrayLike,
    ) -> jnp.ndarray:
        x = q[0]
        dx = q[1]
        th = q[2]
        dth = q[3]
        u = q[4]
        return jnp.cos(th)

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
    f_obj = lambda x: obj_func(
        x,
    )

    return f_ddx, f_ddth, f_obj


def wrap_theta(th: jax.typing.ArrayLike) -> jnp.ndarray:
    th = th % (-2 * np.pi)
    return th


def mpc_test(argv=None):
    def create_environment(
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Env:
        """Creates an environment from the registry.
        Args:
            episode_length: length of episode
            action_repeat: how many repeated actions to take per environment step
            auto_reset: whether to auto reset the environment after an episode is done
            batch_size: the number of environments to batch together
            **kwargs: keyword argments that get passed to the Env class constructor
        Returns:
            env: an environment
        """
        env = cartpole.CartPole(**kwargs)

        if episode_length is not None:
            env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
        if batch_size:
            env = wrapper.VmapWrapper(env, batch_size)
        if auto_reset:
            env = custom_wrapper.AutoResetWrapper(env)

        return env

    # QP Hyperparameters:
    time_horizon = 0.5
    nodes = 11
    num_states = 5

    # Setup QP:
    equaility_functions, inequality_functions, objective_functions, linearized_dynamics = (
        qp_preprocess(
            time_horizon=time_horizon,
            nodes=nodes,
            num_states=num_states,
            mass_cart=1.,
            mass_pole=1.,
            length=0.1,
            gravity=9.81,
        )
    )

    # Isolate Function:
    solve_qp = lambda x, y, z: qp_layer(
        x,
        y,
        z,
        equaility_functions,
        inequality_functions,
        objective_functions,
        linearized_dynamics,
        nodes,
        num_states,
    )

    # Create Environment:
    episode_length = 100
    env = create_environment(
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=1,
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initialize RNG Keys:
    key_seed = 42
    initial_key = jax.random.PRNGKey(key_seed)
    key, env_key = jax.random.split(initial_key)

    # Run Simulation:
    states = reset_fn(env_key)
    state_history = []
    state_history.append(states)
    target = jnp.array([-jnp.pi], dtype=jnp.float64)
    initial_condition = np.squeeze(states.obs)
    initial_condition = np.array([
        initial_condition[0],
        initial_condition[2],
        initial_condition[1],
        initial_condition[3],
    ])
    previous_trajectory = np.repeat(
        a=np.expand_dims(
            np.concatenate([initial_condition, np.array([0.0])]),
            axis=0,
        ),
        repeats=nodes,
        axis=0,
    )
    for _ in range(episode_length):
        key, env_key = jax.random.split(env_key)
        state_trajectory, objective_value, status = solve_qp(
            initial_condition,
            target,
            previous_trajectory,
        )
        previous_trajectory = np.reshape(state_trajectory, (-1, num_states))
        # What is going on here?
        action = np.expand_dims(
            np.expand_dims(state_trajectory[-1, 0], axis=0),
            axis=0,
        )

        # Make sure the QP Layer is solving:
        assert status == 1

        states = step_fn(
            states,
            action,
            env_key,
        )
        initial_condition = np.squeeze(states.obs)
        initial_condition = np.array([
            initial_condition[0],
            initial_condition[2],
            initial_condition[1],
            initial_condition[3],
        ])
        state_history.append(states)

    visualizer.generate_batch_video(
        env=env, states=state_history, batch_size=1, name="cartpole.mp4"
    )


if __name__ == '__main__':
    app.run(mpc_test)
