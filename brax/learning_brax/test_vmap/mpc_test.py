from absl import app
from typing import Optional
import functools

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.envs import wrapper
from brax.envs.env import Env

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm

import qp
import puck
import custom_wrapper

import time


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
    video_length = len(states)
    with writer_obj.saving(fig, name + ".mp4", 300):
        for simulation_step in tqdm(range(0, video_length)):
            fig, writer_obj, puck_patch = _visualize_batch(
                fig=fig,
                writer_obj=writer_obj,
                patches=puck_patches,
                state=states[simulation_step],
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
                state[state_iter][0] - width / 2,
                - height / 2,
            ),
        )
        state_iter += 1
    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patches


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
    env = puck.Puck(**kwargs)

    if episode_length is not None:
        env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrapper.VmapWrapper(env, batch_size)
    if auto_reset:
        env = custom_wrapper.AutoResetWrapper(env)

    return env


@functools.partial(jax.jit, static_argnames=['qp_func', 'reset_fn', 'step_fn', 'episode_length'])
def rollout(
    target_position,
    key,
    qp_func,
    reset_fn,
    step_fn,
    episode_length,
):
    """ Rollout of Environment Episode """
    # Generate Rollout RNG Keys:
    key, reset_key, env_key = jax.random.split(key, 3)
    # Reset Environment:
    states = reset_fn(reset_key)

    # jax.lax.scan function:
    def episode_loop(carry, data):
        # Unpack Carry Tuple:
        states, env_key = carry
        key, env_key = jax.random.split(env_key)
        # Brax Environment Step:
        pos, vel, acc, status = qp_func(states.obs, target_position)
        action = jnp.expand_dims(acc[..., 0], axis=-1)
        states = step_fn(
            states,
            action,
            env_key,
        )

        carry = (states, env_key)
        data = states
        return carry, data

    # Scan over episode:
    carry, data = jax.lax.scan(
        episode_loop,
        (states, env_key),
        (),
        episode_length,
    )

    return carry, data


def main(argv=None):
    # RNG Key:
    key_seed = 42

    # Setup Gym Environment:
    num_envs = 32
    max_episode_length = 200
    env = create_environment(
        episode_length=max_episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Optimization Parameters: (These really matter for solve convergence)
    # time_horizon = 1.0
    # nodes = 11
    time_horizon = 1.0
    nodes = 3

    # Preprocess QP:
    equaility_functions, inequality_functions, objective_functions = qp.qp_preprocess(
        time_horizon,
        nodes,
    )
    # Isolate Function w/ Lambda Function:
    vqp = lambda x, y: qp.qp_layer(
        x, y, equaility_functions, inequality_functions, objective_functions, nodes,
    )

    vqp_layer = jax.vmap(
        vqp,
        in_axes=(0, 0),
        out_axes=(0, 0, 0, 0),
    )

    # Initialize RNG:
    initial_key = jax.random.PRNGKey(key_seed)
    key, subkey = jax.random.split(initial_key)

    # Random Target Positions:
    target_position = []
    for _ in range(num_envs):
        key, subkey = jax.random.split(subkey)
        target_position.append(
            jax.random.uniform(
                key=subkey,
                shape=(1,),
                minval=jnp.array([1.0]),
                maxval=jnp.array([2.0]),
                dtype=jnp.float32,
            ),
        )

    target_position = jnp.asarray(target_position)

    # MPC Loop:
    key, env_key = jax.random.split(subkey)

    # Unrolled Loop:
    _, states = rollout(target_position, key, vqp_layer, reset_fn, step_fn, max_episode_length)

    start_time = time.time()
    _, states = rollout(target_position, key, vqp_layer, reset_fn, step_fn, max_episode_length)
    
    # # For Loop:
    # states = reset_fn(env_key)
    # pos, vel, acc, status = vqp_layer(states.obs, target_position)
    # states_episode = []
    # print('Loop Start')
    # start_time = time.time()
    # for mpc_iteration in range(max_episode_length):
    #     key, env_key = jax.random.split(env_key)
    #     # Brax Environment Step:
    #     pos, vel, acc, status = vqp_layer(states.obs, target_position)
    #     assert (status.status).any()
    #     action = jnp.expand_dims(acc[..., 0], axis=-1)
    #     states = step_fn(
    #         states,
    #         action,
    #         env_key,
    #     )
    #     states_episode.append(states.obs)

    print(f'Elapsed Time: {time.time() - start_time} seconds')

    # generate_batch_video(
    #     states=states_episode,
    #     target=target_position,
    #     batch_size=num_envs,
    #     dt=env.dt,
    #     name=f'./videos/puck_simulation_v2'
    # )


if __name__ == '__main__':
    app.run(main)
