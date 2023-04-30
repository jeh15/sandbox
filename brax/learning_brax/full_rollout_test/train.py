from absl import app
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.envs import wrapper
from brax.envs.env import Env


import model
import model_utilities
import train_utilities
import cartpole
import visualize_cartpole as visualizer
import custom_wrapper

import time


def create_environment(
    environment: callable,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
    """Creates an environment from the registry.
    Args:
        environment: brax environment pipeline object
        episode_length: length of episode
        action_repeat: how many repeated actions to take per environment step
        auto_reset: whether to auto reset the environment after an episode is done
        batch_size: the number of environments to batch together
        **kwargs: keyword argments that get passed to the Env class constructor
    Returns:
        env: an environment
    """
    env = environment(**kwargs)

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


def create_optimizer():
    """Creates a custom optimizer."""
    tx = optax.adam(
        learning_rate=optax.linear_schedule(
        init_value=1e-3,
        end_value=1e-4,
        transition_steps=100,
        transition_begin=300,
        ),
    )
    return tx

def create_train_state(module, params, tx):
    """Creates an initial `TrainState`."""
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
    num_envs = 256
    max_episode_length = 500
    epsilon = 0.0
    reward_threshold = max_episode_length - epsilon
    training_length = 6000
    env_hv = create_environment(
        environment=cartpole.CartPoleHV,
        episode_length=max_episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
    )
    step_fn_hv = jax.jit(env_hv.step)
    reset_fn_hv = jax.jit(env_hv.reset)
    env_lv = create_environment(
        environment=cartpole.CartPoleLV,
        episode_length=max_episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
    )
    step_fn_lv = jax.jit(env_lv.step)
    reset_fn_lv = jax.jit(env_lv.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)
    network = model.ActorCriticNetwork(action_space=env_hv.num_actions)
    initial_params = init_params(
        module=network,
        input_size=env_hv.observation_size,
        key=initial_key,
    )
    # Create a train state:
    tx = create_optimizer()
    model_state = create_train_state(
        module=network,
        params=initial_params,
        tx=tx,
    )
    del initial_params

    # Environment Parameters:
    iteration_schedule = 300
    step_fn = step_fn_hv
    reset_fn = reset_fn_hv

    # Training Loop:
    average_reward = 0.0
    convergence_counter = 0
    key, subkey = jax.random.split(initial_key)
    for iteration in range(training_length):
        # After iteration schedule switch hyperparameters:
        if iteration == iteration_schedule:
            step_fn = step_fn_lv
            reset_fn = reset_fn_lv

        key, subkey = jax.random.split(subkey)
        model_state, loss, carry, data = train_utilities.rollout(
            model_state,
            subkey,
            reset_fn,
            step_fn,
            max_episode_length,
        )

        # Unpack data:
        states_episode, values_episode, log_probability_episode, \
            actions_episode, rewards_episode, masks_episode = data

        previous_reward = average_reward
        average_reward = np.mean(
            np.sum(
                (rewards_episode),
                axis=0,
            ),
        )

        if average_reward >= best_reward:
            best_reward = average_reward
            best_iteration = iteration

        if iteration % 5 == 0:
            print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss}')

        if average_reward >= reward_threshold:
            if average_reward == previous_reward:
                convergence_counter += 1
            else:
                convergence_counter = 0

            if convergence_counter == 5:
                print(f'Reward threshold achieved at iteration {iteration}')
                print(f'Average Reward: {average_reward} \t Loss: {loss}')
                break

    if convergence_counter != 5:
        print(f'The algorithm did not converge to a solution.')
        print(f'The best reward of {best_reward} was achieved at iteration {best_iteration}')

    state_history = []
    states = reset_fn(subkey)
    state_history.append(states)
    for _ in range(max_episode_length):
        key, subkey = jax.random.split(subkey)
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
                subkey,
            )
        )
        states = jax.lax.stop_gradient(
            step_fn(
                states,
                actions,
                subkey,
            )
        )
        state_history.append(states)

    visualize_batches = 25
    visualizer.generate_batch_video(
        env=env_lv,
        states=state_history,
        batch_size=visualize_batches,
        name=f'./videos/cartpole_simulation_{iteration}'
    )


if __name__ == '__main__':
    app.run(main)
