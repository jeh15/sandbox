from absl import app

from clu import metrics
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
import optax
import gymnasium as gym

import actor
import critic
import model_utilities

import pdb


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
    env = gym.make('CartPole-v1')
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

    print(actor_network.tabulate(jax.random.PRNGKey(0), jnp.ones(env.observation_space.shape)))
    print(critic_network.tabulate(jax.random.PRNGKey(0), jnp.ones(env.observation_space.shape)))

    # Create a train state:
    learning_rate = 0.01
    momentum = 0.9
    actor_state = create_train_state(
        module=actor_network,
        params=actor_params,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    critic_state = create_train_state(
        module=critic_network,
        params=critic_params,
        learning_rate=learning_rate,
        momentum=momentum,
    )

    # Train on Random Data:
    key = jax.random.PRNGKey(0)
    epochs = 100
    for iteration in range(epochs):
        num_batch = 1
        batch_agent_states = jax.random.uniform(key=key, shape=[num_batch, 6])
        batch_returns = jax.random.uniform(key=key, shape=[num_batch, 1])
        state, loss = train_step(state, batch_agent_states, batch_returns)
        if iteration % 10 == 0:
            print(f'Iteration: {iteration}, Loss: {loss}')


if __name__ == '__main__':
    app.run(main)
