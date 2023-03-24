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

import model
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
    env = gym.make(
        'CartPole-v1',
        render_mode="rgb_array",
        max_episode_steps=500,
    )
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        episode_trigger=lambda x: x % 100 == 0,
    )
    initial_key = jax.random.PRNGKey(42)
    # Initize Networks:
    network = model.ActorCriticNetwork(action_space=env.action_space.n)
    initial_params = init_params(
        module=network,
        input_size=env.observation_space.shape,
        key=initial_key,
    )
    # Create a train state:
    learning_rate = 2.5e-4
    model_state = create_train_state(
        module=network,
        params=initial_params,
        learning_rate=learning_rate,
    )
    del initial_params

    # Test Environment:
    epochs = 1001
    key, subkey = jax.random.split(initial_key)
    loss_history = []
    for iteration in range(epochs):
        states = env.reset()[0]
        reset_flag = False
        values_episode = []
        states_episode = []
        rewards_episode = []
        key, subkey = jax.random.split(subkey)
        while not reset_flag:
            logits, values = model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                states,
            )
            actions, log_probability, entropy = model_utilities.select_action(
                logits,
                subkey,
            )
            states, rewards, terminated, truncated, infos = env.step(
                action=np.array(actions),
            )
            if not (terminated or truncated):
                values_episode.append(values)
                states_episode.append(states)
                rewards_episode.append(rewards)
            else:
                values_episode.append(values)
                states_episode.append(states)
                rewards_episode.append(-1.0)
                reset_flag = True

        values_episode = jnp.asarray(values_episode, dtype=jnp.float32)
        states_episode = jnp.asarray(states_episode, dtype=jnp.float32)
        rewards_episode = jnp.asarray(rewards_episode, dtype=jnp.float32)

        advantage_episode = model_utilities.calculate_advantage(
            rewards_episode,
            values_episode,
        )

        # Update Function:
        model_state, loss = model_utilities.train_step(
            model_state,
            advantage_episode,
            states_episode,
            subkey,
        )

        if iteration % 10 == 0:
            print(f'Epoch: {iteration} \t Reward: {np.sum(rewards_episode)} \t Loss: {loss}')

        loss_history.append(loss)

    env.close()

    # Plot Results:
    fig, ax = plt.subplots()
    fig.tight_layout(pad=2.5)
    loss_plot, = ax.plot(loss_history, color='cornflowerblue', alpha=0.5, linewidth=1.0)
    plt.show()
    plt.savefig("loss_plot.png")


if __name__ == '__main__':
    app.run(main)
