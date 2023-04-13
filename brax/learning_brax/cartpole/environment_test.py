from absl import app
from typing import Optional

import jax
import jax.numpy as jnp
from brax.envs import wrapper
from brax.envs.env import Env

import cartpole
import visualize_cartpole as visualizer


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
        env = wrapper.AutoResetWrapper(env)

    return env


def main(argv=None):
    # RNG Key:
    key_seed = 0
    key = jax.random.PRNGKey(key_seed)

    training_length = 5000

    env = create_environment(
        episode_length=1000,
        action_repeat=1,
        auto_reset=True,
    )

    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    state = reset_fn(rng=key)
    states = []
    for iteration in range(training_length):
        state = step_fn(state, jnp.array([0], dtype=jnp.float32))
        states.append(state)

    visualizer.generate_video(env=env, states=states)


if __name__ == '__main__':
    app.run(main)
