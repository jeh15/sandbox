from absl import app
from typing import Optional

import jax
import jax.numpy as jnp
from brax.envs import wrapper
from brax.envs.env import Env

import cartpole
import visualize_cartpole as visualizer
import custom_wrapper

import time


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


def main(argv=None):
    # Parameters:
    training_length = 1000
    batch_size = 256

    # RNG Key:
    key_seed = 0
    key = jax.random.PRNGKey(key_seed)

    env = create_environment(
        episode_length=1000,
        action_repeat=1,
        auto_reset=True,
        batch_size=batch_size,
    )

    t = time.perf_counter()
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)
    elapsed_time = time.perf_counter() - t
    print(f'JIT Compile Time: {elapsed_time:.2f} s')

    state = reset_fn(rng=key)
    states = []
    t = time.perf_counter()
    zero_action = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    for iteration in range(training_length):
        key, subkey = jax.random.split(key)
        random_action = jax.random.uniform(
            key=subkey, shape=(batch_size, 1), dtype=jnp.float32, minval=-1, maxval=1,
        )
        state = step_fn(
            state, random_action, subkey,
        )
        states.append(state)
    elapsed_time = time.perf_counter() - t
    print(f'Simulation Time: {elapsed_time:.2f} s for {batch_size} simulations')

    visualize_batches = 4
    visualizer.generate_batch_video(
        env=env, states=states, batch_size=visualize_batches,
    )


if __name__ == '__main__':
    app.run(main)
