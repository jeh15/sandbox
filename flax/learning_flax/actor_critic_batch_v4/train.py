import os
from absl import app

from clu import metrics
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
import optax
import gymnasium as gym
import matplotlib.pyplot as plt

import policy_model
import value_model
import model_utilities

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def init_params(module, input_size, key):
    params = module.init(
        key,
        jnp.ones(input_size),
    )['params']
    return params


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, params, learning_rate):
    """Creates an initial `TrainState`."""
    tx = optax.adam(
        learning_rate=learning_rate,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
    )


def main(argv=None):
    # Initialize Environment:
    epochs = 1001
    num_batch = 16
    num_steps = 200
    sample_rate = 500
    env = gym.make(
        'CartPole-v1',
        render_mode="rgb_array",
        max_episode_steps=num_steps,
    )
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        episode_trigger=lambda x: x % (num_batch * sample_rate) == 0,
    )
    initial_key = jax.random.PRNGKey(42)
    # Initize Networks:
    actor_network = policy_model.PolicyNetwork(action_space=env.action_space.n)
    critic_network = value_model.ValueNetwork()
    actor_initial_params = init_params(
        module=actor_network,
        input_size=env.observation_space.shape,
        key=initial_key,
    )
    critic_initial_params = init_params(
        module=critic_network,
        input_size=env.observation_space.shape,
        key=initial_key,
    )
    # Create a train state:
    learning_rate = 0.001
    actor_model_state = create_train_state(
        module=actor_network,
        params=actor_initial_params,
        learning_rate=learning_rate,
    )
    critic_model_state = create_train_state(
        module=critic_network,
        params=critic_initial_params,
        learning_rate=learning_rate,
    )
    del actor_initial_params, critic_initial_params

    # Test Environment:
    key, subkey = jax.random.split(initial_key)
    for iteration in range(epochs):
        values_episode = np.zeros(
            (num_batch, num_steps, 1),
            dtype=np.float32,
        )
        states_episode = np.zeros(
            (num_batch, num_steps, env.observation_space.shape[0]),
            dtype=np.float32,
        )
        rewards_episode = np.zeros(
            (num_batch, num_steps, 1),
            dtype=np.float32,
        )
        masks_episode = np.zeros(
            (num_batch, num_steps, 1),
            dtype=np.int16,
        )
        random_flag_episode = []
        for batch in range(num_batch):
            states = env.reset()[0]
            reset_flag = False
            key, subkey = jax.random.split(subkey)
            random_flag = 0 if jax.random.uniform(subkey) > 0.1 else 1
            random_flag_episode.append(random_flag)
            step_iterator = 0
            while not reset_flag:
                logits = model_utilities.actor_forward_pass(
                    actor_model_state.params,
                    actor_model_state.apply_fn,
                    states,
                )
                values = model_utilities.critic_forward_pass(
                    critic_model_state.params,
                    critic_model_state.apply_fn,
                    states,
                )
                if not random_flag:
                    actions, log_probability, entropy = model_utilities.select_action(
                        logits,
                        subkey,
                    )
                else:
                    actions = env.action_space.sample()

                next_states, rewards, terminated, truncated, infos = env.step(
                    action=np.array(actions),
                )
                if not (terminated or truncated):
                    values_episode[batch, step_iterator, :] = values
                    states_episode[batch, step_iterator, :] = states
                    rewards_episode[batch, step_iterator, :] = rewards
                    masks_episode[batch, step_iterator, :] = 1
                else:
                    values_episode[batch, step_iterator:, :] = values
                    states_episode[batch, step_iterator:, :] = states
                    rewards_episode[batch, step_iterator:, :] = 0
                    masks_episode[batch, step_iterator:, :] = 0
                    reset_flag = True
                states = next_states
                step_iterator += 1

        values_episode = jnp.asarray(values_episode)
        states_episode = jnp.asarray(states_episode)
        rewards_episode = jnp.asarray(rewards_episode)
        masks_episode = jnp.asarray(masks_episode)

        advantage_episode, returns_episode = model_utilities.calculate_advantage(
            rewards_episode,
            values_episode,
            masks_episode,
            num_steps,
        )

        # Update Function:
        actor_model_state, critic_model_state, policy_loss, value_loss = model_utilities.train_step(
            actor_model_state,
            critic_model_state,
            advantage_episode,
            returns_episode,
            states_episode,
            subkey,
            num_batch,
            num_steps,
        )

        if iteration % 5 == 0:
            print(f'Epoch: {iteration} \t Number of Random Policies: {np.sum(random_flag_episode)} \t Average Reward: {np.mean(np.sum(rewards_episode, axis=1))} \t Policy Loss: {policy_loss} \t Value Loss: {value_loss}')


    env.close()


if __name__ == '__main__':
    app.run(main)
