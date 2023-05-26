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


import model_no_qp as model
import model_utilities_no_qp as model_utilities
import puck
import custom_wrapper
import visualize_puck as visualizer
import save_checkpoint


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
    num_envs = 1
    max_episode_length = 100
    training_length = 500
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
        states = reset_fn(env_key)
        state_history = []
        state_history.append(states)
        states_episode = []
        values_episode = []
        log_probability_episode = []
        actions_episode = []
        rewards_episode = []
        masks_episode = []
        start_time = time.time()
        for environment_step in range(max_episode_length):
            # Brax Environment Step:
            key, env_key = jax.random.split(env_key)
            mean, std, values = model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states.obs,
            )
            actions, log_probability, entropy = model_utilities.select_action(
                mean,
                std,
                env_key,
            )
            next_states = step_fn(
                states,
                actions,
                env_key,
            )
            states_episode.append(states.obs)
            values_episode.append(jnp.squeeze(values))
            log_probability_episode.append(jnp.squeeze(log_probability))
            actions_episode.append(jnp.squeeze(actions))
            rewards_episode.append(jnp.squeeze(states.reward))
            masks_episode.append(jnp.where(states.done == 0, 1.0, 0.0))
            states = next_states
            state_history.append(states)

        # Convert to Jax Arrays:
        states_episode = jnp.swapaxes(
            jnp.asarray(states_episode), axis1=1, axis2=0,
        )
        values_episode = jnp.swapaxes(
            jnp.asarray(values_episode), axis1=1, axis2=0,
        )
        log_probability_episode = jnp.swapaxes(
            jnp.asarray(log_probability_episode), axis1=1, axis2=0,
        )
        actions_episode = jnp.swapaxes(
            jnp.asarray(actions_episode), axis1=1, axis2=0,
        )
        rewards_episode = jnp.swapaxes(
            jnp.asarray(rewards_episode), axis1=1, axis2=0,
        )
        masks_episode = jnp.swapaxes(
            jnp.asarray(masks_episode), axis1=1, axis2=0,
        )

        # No Gradient Calculation:
        _, _, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states.obs,
            ),
        )

        # Calculate Advantage:
        values_episode = jnp.concatenate(
            [values_episode, values],
            axis=1,
        )

        advantage_episode, returns_episode = jax.lax.stop_gradient(
            model_utilities.calculate_advantage(
                rewards_episode,
                values_episode,
                masks_episode,
                max_episode_length,
            )
        )

        # Update Function:
        model_state, loss = model_utilities.train_step(
            model_state,
            states_episode,
            actions_episode,
            advantage_episode,
            returns_episode,
            log_probability_episode,
        )

        average_reward = np.mean(
            np.sum(
                (rewards_episode),
                axis=0,
            ),
        )

        if average_reward >= best_reward:
            best_reward = average_reward
            best_iteration = iteration

        # if iteration % 25 == 0:
        #     visualize_batches = 9
        #     visualizer.generate_batch_video(
        #         env=env,
        #         states=state_history,
        #         batch_size=visualize_batches,
        #         name=f'./videos/puck_training_{iteration}'
        #     )

        print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss} \t Elapsed Time: {time.time() - start_time}')

    print(f'The best reward of {best_reward} was achieved at iteration {best_iteration}')

    # Checkpoint Model:
    save_checkpoint.save_checkpoint(state=model_state, path='./checkpoints')


if __name__ == '__main__':
    app.run(main)
