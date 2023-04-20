from absl import app
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from brax.envs import wrapper
from brax.envs.env import Env


import model
import model_utilities
import train_utilities
import cartpole
import visualize_cartpole as visualizer
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
    env = cartpole.CartPole(**kwargs)

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
    checkpoint_flag = False

    # Setup Gym Environment:
    num_envs = 512
    max_episode_length = 500
    epsilon = 0.0
    reward_threshold = max_episode_length - epsilon
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
    network = model.ActorCriticNetwork(action_space=env.num_actions)
    initial_params = init_params(
        module=network,
        input_size=env.observation_size,
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
    key, subkey = jax.random.split(initial_key)
    for iteration in range(training_length):
        key, subkey = jax.random.split(subkey)
        carry, data = train_utilities.rollout(
            model_state,
            subkey,
            reset_fn,
            step_fn,
            max_episode_length,
        )
        # Unpack carry and data:
        model_state, states, env_key = carry
        states_episode, values_episode, log_probability_episode, \
            actions_episode, rewards_episode, masks_episode = data

        # No Gradient Calculation:
        _, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states.obs,
            ),
        )

        # Calculate Advantage:
        values_episode = jnp.concatenate(
            [values_episode, jnp.expand_dims(np.squeeze(values), axis=0)],
            axis=0,
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
                (rewards_episode * masks_episode),
                axis=0,
            ),
        )

        if iteration % 5 == 0:
            print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss}')

        if average_reward >= reward_threshold:
            print(f'Reward threshold achieved at iteration {iteration}')
            break

        # if iteration % 50 == 0:
        #     visualize_batches = 4
        #     visualizer.generate_batch_video(
        #         env=env,
        #         states=state_history,
        #         batch_size=visualize_batches,
        #         name=f'./videos/cartpole_simulation_{iteration}'
        #     )

    # if checkpoint_flag:
    #     CKPT_DIR = './checkpoints'
    #     checkpoints.save_checkpoint(
    #         ckpt_dir=CKPT_DIR,
    #         target=model_state,
    #         step=iteration,
    #     )

    # Replay Policy:
    env = create_environment(
        episode_length=max_episode_length,
        action_repeat=1,
    )
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    states = reset_fn(subkey)
    state_history = []
    state_history.append(states)
    for step in range(int(max_episode_length - epsilon)):
        key, subkey = jax.random.split(subkey)
        logits, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states.obs,
            )
        )
        actions, log_probability, entropy = jax.lax.stop_gradient(
            model_utilities.select_action(
                logits,
                subkey,
            )
        )
        actions = jnp.expand_dims(
            model_utilities.map_action(actions),
            axis=-1,
        )
        states = jax.lax.stop_gradient(
            step_fn(
                states,
                actions,
                subkey,
            )
        )
        state_history.append(states)

    visualizer.generate_video(
        env=env,
        states=state_history,
        name=f'./videos/cartpole_learned_policy'
    )


if __name__ == '__main__':
    app.run(main)