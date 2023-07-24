from typing import Optional
import os
from absl import app

import jax
import jax.numpy as jnp
import brax
from brax.io import mjcf
from brax.positional import pipeline
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env
from tqdm import tqdm

import unitree_a1
import visualize
import custom_wrapper


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
    env = unitree_a1.unitree_a1(**kwargs)

    if episode_length is not None:
        env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrapper.VmapWrapper(env, batch_size)
    if auto_reset:
        env = custom_wrapper.AutoResetWrapper(env)

    return env


def main(argv=None):
    # Import mjcf file:
    filename = "models/unitree/scene.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        filename,
    )

    env = create_environment(
        episode_length=None,
        auto_reset=False,
        batch_size=False,
        backend='generalized',
    )

    # Load mjcf model:
    pipeline_model = mjcf.load(filepath)
    motor_mask = pipeline_model.actuator.qd_id

    initial_q = jnp.array(
        [0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8,
         0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8],
        dtype=jnp.float32,
    )
    base_ctrl = jnp.array(
        [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8],
        dtype=jnp.float32,
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    initial_key = jax.random.PRNGKey(42)
    state = reset_fn(initial_key)

    state_history = [state.pipeline_state]
    simulation_steps = 200
    for i in tqdm(range(simulation_steps)):
        ctrl_input = jnp.zeros((base_ctrl.shape[0],), dtype=jnp.float32)
        front_left = jnp.array([0, jnp.sin(i / 50), 0])
        front_right = -jnp.array([0, jnp.sin(i / 50), 0])
        back_left = -jnp.array([0, jnp.sin(i / 50), 0])
        back_right = jnp.array([0, jnp.sin(i / 50), 0])
        sine_force = 5 * jnp.concatenate([front_right, front_left, back_right, back_left])
        state = step_fn(state, sine_force)
        state_history.append(state.pipeline_state)

    video_filepath = os.path.join(os.path.dirname(__file__), "unitree_simulation")
    visualize.create_video(
        sys=pipeline_model,
        states=state_history,
        width=1280,
        height=720,
        name="Unitree a1",
        filepath=video_filepath,
    )


if __name__ == "__main__":
    app.run(main)
