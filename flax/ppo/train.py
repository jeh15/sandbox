from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import gymnasium as gym

import model
import model_utilities


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
                    disable_logger=True,
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

    # Setup Gym Environment:
    num_envs = 512
    max_episode_length = 500
    epsilon = 0.1
    reward_threshold = max_episode_length - epsilon
    training_length = 1000
    video_rate = 500
    envs = gym.vector.SyncVectorEnv(
        [make_environment(key_seed + i, i, max_episode_length=max_episode_length, video_rate=video_rate, video_enable=False) for i in range(num_envs)]
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
