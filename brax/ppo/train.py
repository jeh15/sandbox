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
    epsilon = 0.1
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
    network = model.ActorCriticNetwork(action_space=envs.single_action_space.n)
    initial_params = init_params(
        module=network,
        input_size=envs.single_observation_space.shape[0],
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
    key, reset_key, env_key = jax.random.split(initial_key, 3)
    for iteration in range(training_length):
        states_episode = np.zeros(
            (max_episode_length, num_envs, 4),
            dtype=np.float32,
        )
        values_episode = np.zeros(
            (max_episode_length+1, num_envs),
            dtype=np.float32,
        )
        log_probability_episode = np.zeros(
            (max_episode_length, num_envs),
            dtype=np.float32,
        )
        actions_episode = np.zeros(
            (max_episode_length, num_envs),
            dtype=np.float32,
        )
        rewards_episode = np.zeros(
            (max_episode_length, num_envs),
            dtype=np.float32,
        )
        masks_episode = np.zeros(
            (max_episode_length, num_envs),
            dtype=np.int16,
        )
        # Reset all environments:
        key, reset_key = jax.random.split(reset_key)
        states = reset_fn(subkey)
        for step in range(max_episode_length):
            # Provide new keys to auto-reset environments:
            key, env_key = jax.random.split(env_key)
            logits, values = model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states,
            )
            actions, log_probability, entropy = model_utilities.select_action(
                logits,
                env_key,
            )
            next_states = step_fn(
                states,
                actions,
                env_key,
            )
            states_episode[step] = states
            values_episode[step] = np.squeeze(values)
            log_probability_episode[step] = np.squeeze(log_probability)
            actions_episode[step] = np.squeeze(actions)
            rewards_episode[step] = np.squeeze(rewards)
            masks_episode[step] = np.array([not terminate for terminate in terminated])
            states = next_states

        # No Gradient Calculation:
        _, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states,
            ),
        )
        values_episode[-1] = np.squeeze(values)
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

    envs_wrapper.close()

    if checkpoint_flag:
        CKPT_DIR = './checkpoints'
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR,
            target=model_state,
            step=iteration,
        )

    # Record Learned Policy:
    env = gym.make(
            'CartPole-v1',
            render_mode="rgb_array",
            max_episode_steps=max_episode_length,
    )
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        episode_trigger=lambda x: x == 0,
        name_prefix='pretrained-replay',
        disable_logger=True,
    )

    states, info = env.reset(seed=key_seed)
    terminated = 0
    truncated = 0
    while not (terminated or truncated):
        key, subkey = jax.random.split(subkey)
        logits, _ = model_utilities.forward_pass(
            model_state.params,
            model_state.apply_fn,
            states,
        )
        actions, _, _ = model_utilities.select_action(
            logits,
            subkey,
        )
        states, _, terminated, truncated, _ = env.step(
            action=np.array(actions),
        )

    env.close()


if __name__ == '__main__':
    app.run(main)
