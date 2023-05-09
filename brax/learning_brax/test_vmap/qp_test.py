from functools import partial
from typing import Callable

from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm


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

    # 1. Initial Condition Constraints:
    initial_condition = jnp.asarray([
        x[0] - initial_conditions[..., 0],
        dx[0] - initial_conditions[..., 1],
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
    position_limit = 2.4
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
    target_objective = 10 * jnp.sum((x - target_position) ** 2, axis=0)
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
        primal_infeasible_tol=1e-3,
        dual_infeasible_tol=1e-3,
        rho_start=1e-2,
        maxiter=8000,
        tol=1e-3,
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
    return pos, vel, acc, state


def generate_batch_video(
        states: jax.typing.ArrayLike,
        target: jax.typing.ArrayLike,
        batch_size: int,
        dt: float,
        name: str,
):
    # Subplot Layout: (Finds closest square)
    layout = np.floor(
        np.sqrt(batch_size)
    ).astype(int)

    # Create plot handles for visualization:
    fig, axes = plt.subplots(nrows=layout, ncols=layout)

    lb, ub = -2.4, 2.4
    for ax in axes.flatten():
        ax.axis('equal')
        ax.set_xlim([lb, ub])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Simulation:')

    # Initialize Patch: (puck)
    width = 1.0 / 2
    height = 0.5 / 2
    xy_puck = (0, 0)
    puck_patches = []
    goal_patches = []
    for iteration in range(batch_size):
        goal_patch = Circle(
            (target[iteration], 0.25), radius=0.1, color='red', zorder=15,
        )
        puck_patch = Rectangle(
            xy_puck, width, height, color='cornflowerblue', zorder=5,
        )
        puck_patches.append(puck_patch)
        goal_patches.append(goal_patch)

    iteration = 0
    for ax, puck_patch, goal_patch in zip(axes.flatten(), puck_patches, goal_patches):
        ax.text(
            target[iteration],
            0.6,
            'Goal',
            fontsize=6,
            horizontalalignment='center',
            verticalalignment='center',
        )
        ax.add_patch(puck_patch)
        ax.add_patch(goal_patch)
        ax.hlines(0, lb, ub, colors='black', linewidth=0.75, linestyles='--', zorder=0)
        iteration += 1

    # Create video writer:
    fps = 24
    rate = int(1.0 / (dt * fps))
    rate = rate if rate >= 1 else 1
    writer_obj = FFMpegWriter(fps=fps)
    video_length = states.shape[-1]
    with writer_obj.saving(fig, name + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length)):
            fig, writer_obj, puck_patch = _visualize_batch(
                fig=fig,
                writer_obj=writer_obj,
                patches=puck_patches,
                state=states[..., simulation_step],
                width=width,
                height=height,
            )


def _visualize_batch(fig, writer_obj, patches, state, width, height):
    puck_patches = patches
    state_iter = 0
    for puck_patch in puck_patches:
        # Update Patch: (x, z) position
        puck_patch.set(
            xy=(
                state[state_iter] - width / 2,
                - height / 2,
            ),
        )
        state_iter += 1
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patches


# Test JAXopt OSQP:
def main(argv=None) -> None:
    # Random Key:
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # Optimization Parameters: (These really matter for solve convergence)
    time_horizon = 5.0
    nodes = 21
    num_optimizations = 25

    # Dummy Inputs:
    initial_condition = []
    target_position = []
    for _ in range(num_optimizations):
        key, subkey = jax.random.split(subkey)
        initial_condition.append(
            jax.random.uniform(
                key=subkey,
                shape=(2,),
                minval=jnp.array([-2.0, 0]),
                maxval=jnp.array([-1.0, 0]),
                dtype=jnp.float32,
            ),
        )
        target_position.append(
            jax.random.uniform(
                key=subkey,
                shape=(1,),
                minval=jnp.array([1.0]),
                maxval=jnp.array([2.0]),
                dtype=jnp.float32,
            ),
        )

    initial_condition = jnp.asarray(initial_condition)
    target_position = jnp.asarray(target_position)

    # Preprocess QP:
    equaility_functions, inequality_functions, objective_functions = qp_preprocess(
        time_horizon,
        nodes,
    )

    # Isolate Function w/ Lambda Function:
    vqp = lambda x, y: qp_layer(
        x, y, equaility_functions, inequality_functions, objective_functions, nodes,
    )
    vqp_layer = jax.vmap(
        vqp,
        in_axes=(0, 0),
        out_axes=(0, 0, 0, 0),
    )

    # Warmup:
    _, _, _, _ = vqp_layer(
        initial_condition,
        target_position,
    )

    # Solve QP:
    start_time = time.time()
    pos, vel, acc, state = vqp_layer(
        initial_condition,
        target_position,
    )
    elapsed_time = time.time() - start_time
    print(f'Elapsed Time: {elapsed_time:.3f} seconds')

    # Print Status:
    print(f'Optimization Solved: {(state.status).any()}')

    visualize_batches = 25
    generate_batch_video(
        states=pos,
        target=target_position,
        batch_size=visualize_batches,
        dt=time_horizon/(nodes - 1),
        name=f'./videos/puck_simulation'
    )


if __name__ == '__main__':
    app.run(main)
