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

import actor
import critic
import model_utilities as model

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def init_params(module, input_size, rng):
    params = module.init(
        rng,
        jnp.ones(input_size),
    )['params']
    return params

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, params, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
    )


def main(argv=None):
    # Initialize Environment:
    env = gym.make('CartPole-v1', render_mode="rgb_array", max_episode_steps=500)
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        episode_trigger=lambda x: x % 200 == 0,
    )
    init_rng = jax.random.PRNGKey(42)
    # Initize Networks:
    actor_network = actor.ActorNetwork(action_space=env.action_space.n)
    actor_params = init_params(
        module=actor_network,
        input_size=env.observation_space.shape,
        rng=init_rng,
    )
    critic_network = critic.CriticNetwork()
    critic_params = init_params(
        module=critic_network,
        input_size=env.observation_space.shape,
        rng=init_rng,
    )
    del init_rng

    # Create a train state:
    actor_lr = 0.001
    critic_lr = 0.005
    momentum = 0.9
    actor_state = create_train_state(
        module=actor_network,
        params=actor_params,
        learning_rate=actor_lr,
        momentum=momentum,
    )
    critic_state = create_train_state(
        module=critic_network,
        params=critic_params,
        learning_rate=critic_lr,
        momentum=momentum,
    )

    # Test Environment:
    epochs = 2001
    key = jax.random.PRNGKey(42)
    actor_loss_history = []
    critic_loss_history = []
    for iteration in range(epochs):
        states = env.reset()[0]
        init_states = states
        reset_flag = False
        reward_episode = []
        states_episode = []
        while not reset_flag:
            key, subkey = jax.random.split(key)
            logits, values = model.forward_pass(
                actor=actor_network,
                critic=critic_network,
                actor_params=actor_state.params,
                critic_params=critic_state.params,
                x=states,
            )
            actions, log_probability, entropy = model.select_action(
                key=subkey,
                logits=logits,
            )
            states, rewards, terminated, truncated, infos = env.step(
                action=np.array(actions),
            )
            if not (terminated or truncated):
                states_episode.append(states)
                reward_episode.append(rewards)
            else:
                reset_flag = True

        # Convert to Jax Array:
        states_episode = jnp.asarray(states_episode, dtype=jnp.float32)
        reward_episode = jnp.asarray(
            reward_episode,
            dtype=jnp.float32,
        ).flatten()

        critic_state, critic_loss = model.update_critic(
            actor_state=actor_state,
            critic_state=critic_state,
            actor_network=actor_network,
            critic_network=critic_network,
            states=states_episode,
            rewards=reward_episode,
            key=subkey,
        )
        actor_state, actor_loss = model.update_actor(
            actor_state=actor_state,
            critic_state=critic_state,
            actor_network=actor_network,
            critic_network=critic_network,
            states=states_episode,
            rewards=reward_episode,
            key=subkey,
        )

        if iteration % 100 == 0:
            print(f'Epoch: {iteration} \t Initial State: {init_states} \t Critic Loss: {critic_loss} \t Actor Loss: {actor_loss}')

        actor_loss_history.append(actor_loss)
        critic_loss_history.append(critic_loss)

    env.close()

    # Plot Results:
    fig, ax = plt.subplots(2)
    fig.tight_layout(pad=2.5)
    critic_plot, = ax[0].plot(critic_loss_history, color='cornflowerblue', alpha=0.5, linewidth=1.0)
    actor_plot, = ax[1].plot(actor_loss_history, color='cornflowerblue', alpha=0.5, linewidth=1.0)
    plt.show()
    plt.savefig("loss_plot.png")


if __name__ == '__main__':
    app.run(main)
