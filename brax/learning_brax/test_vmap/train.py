from absl import app
from typing import Optional
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.envs import wrapper
from brax.envs.env import Env


import model_base as model
import model_utilities
import train_utilities
import puck
import visualize_puck as visualizer
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
    env = puck.Puck(**kwargs)

    if episode_length is not None:
        env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrapper.VmapWrapper(env, batch_size)
    if auto_reset:
        env = custom_wrapper.AutoResetWrapper(env)

    return env


def init_params(module, input_size, key):
    params = module.init(
        key,
        jnp.ones(input_size),
    )['params']
    return params


def create_train_state(module, params, learning_rate):
    """Creates an initial `TrainState`."""
    tx = optax.adam(
        learning_rate=learning_rate,
    )
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
    )


def main(argv=None):
    # RNG Key:
    key_seed = 42

    best_reward = 0.0
    best_iteration = 0

    # Setup Gym Environment:
    num_envs = 32
    max_episode_length = 200
    training_length = 1000
    env = create_environment(
        episode_length=max_episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)
    # Vmap Network:
    network = model.ActorCriticNetworkVmap(
        action_space=env.num_actions,
        time_horizon=0.1,
        nodes=3,
    )
    initial_params = init_params(
        module=network,
        input_size=(num_envs, env.observation_size),
        key=initial_key,
    )

    # Create a train state:
    learning_rate = 0.001
    model_state = create_train_state(
        module=network,
        params=initial_params,
        learning_rate=learning_rate,
    )
    del initial_params

    # Test Environment:
    key, env_key = jax.random.split(initial_key)
    for iteration in range(training_length):
        key, env_key = jax.random.split(env_key)
        start_time = time.time()
        model_state, loss, carry, data = train_utilities.rollout(
            model_state=model_state,
            key=env_key,
            reset_fn=reset_fn,
            step_fn=step_fn,
            episode_length=max_episode_length,
        )

        # Unpack data:
        states_episode, values_episode, log_probability_episode, \
            actions_episode, rewards_episode, masks_episode = data

        average_reward = np.mean(
            np.sum(
                (rewards_episode),
                axis=0,
            ),
        )

        if average_reward >= best_reward:
            best_reward = average_reward
            best_iteration = iteration

        print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss} \t Elapsed Time: {time.time() - start_time}')

    print(f'The best reward of {best_reward} was achieved at iteration {best_iteration}')

    state_history = []
    states = reset_fn(env_key)
    state_history.append(states)
    for _ in range(max_episode_length):
        key, env_key = jax.random.split(env_key)
        mean, std, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states.obs,
            )
        )
        actions, _, _ = jax.lax.stop_gradient(
            model_utilities.select_action(
                mean,
                std,
                env_key,
            )
        )
        states = jax.lax.stop_gradient(
            step_fn(
                states,
                actions,
                env_key,
            )
        )
        state_history.append(states)

    visualize_batches = 25
    visualizer.generate_batch_video(
        env=env,
        states=state_history,
        batch_size=visualize_batches,
        name=f'./videos/puck_simulation_{iteration}'
    )


if __name__ == '__main__':
    app.run(main)
