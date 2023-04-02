import os
from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
import optax
import gymnasium as gym

import model
import model_utilities

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def make_environment(key, index, max_episode_length, video_rate, video_enable):
    def thunk():
        env = gym.make(
            'CartPole-v1',
            render_mode="rgb_array",
            max_episode_steps=max_episode_length,
        )
        if video_enable:
            if index == 0:
                env = gym.wrappers.RecordVideo(
                    env=env,
                    video_folder="./video",
                    episode_trigger=lambda x: x % (video_rate) == 0,
                )
        env.np_random = key
        return env
    return thunk


def init_params(module, input_size, key):
    params = module.init(
        key,
        jnp.ones(input_size),
    )['params']
    return params


class TrainState(train_state.TrainState):
    pass


def create_train_state(module, params, learning_rate):
    """Creates an initial `TrainState`."""
    tx = optax.adam(
        learning_rate=learning_rate,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
    )


def main(argv=None):
    # RNG Key:
    key_seed = 42

    # Setup Gym Environment:
    num_envs = 16
    max_episode_length = 500
    epsilon = 1.0
    reward_threshold = max_episode_length - epsilon
    training_length = 400
    video_rate = 100
    envs = gym.vector.SyncVectorEnv(
        [make_environment(key_seed + i, i, max_episode_length=max_episode_length, video_rate=video_rate, video_enable=True) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

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
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=num_envs * training_length)
    key, subkey = jax.random.split(initial_key)
    for iteration in range(training_length):
        states_episode = np.zeros(
            (max_episode_length, num_envs, 4),
            dtype=np.float32,
        )
        values_episode = np.zeros(
            (max_episode_length+1, num_envs),
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
        states, info = envs_wrapper.reset(seed=key_seed+iteration)
        for step in range(max_episode_length):
            key, subkey = jax.random.split(subkey)
            logits, values = model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states,
            )
            actions, log_probability, entropy = model_utilities.select_action(
                logits,
                subkey,
            )
            next_states, rewards, terminated, truncated, infos = envs_wrapper.step(
                action=np.array(actions),
            )
            states_episode[step] = states
            values_episode[step] = np.squeeze(values)
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
            max_episode_length,
        )

        average_reward = np.mean(
            np.sum(
                (rewards_episode * masks_episode),
                axis=0,
            ),
        )

        if iteration % 5 == 0:
            print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss}')

    envs_wrapper.close()


if __name__ == '__main__':
    app.run(main)
