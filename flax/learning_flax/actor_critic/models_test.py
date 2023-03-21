from absl import app

from clu import metrics
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
import optax

import sandbox.flax.learning_flax.actor_critic.model_utilities as model_utilities

import pdb

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(
        rng,
        jnp.ones([1, 6]),
        )['params']
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
    )


def train_step(model_state, agent_state, returns):
    """Train for a single step.
    Args::
        model_state: Trainmodel_state class for model
        returns: Batched expected return from environment
    """
    def loss_fn(params, agent_state, returns):
        policy_probabilites, value = model_state.apply_fn(
            {'params': params},
            agent_state,
        )
        advantage = returns - value
        actor_loss = -jnp.sum(
            jnp.log(policy_probabilites) * advantage
        )
        critic_loss = optax.huber_loss(
            predictions=value,
            targets=returns,
            delta=1.0,
        )
        loss = jnp.mean(actor_loss + critic_loss)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(model_state.params, agent_state, returns)
    model_state = model_state.apply_gradients(grads=grads)
    return model_state, loss


def main(argv=None):
    mlp = model_utilities.ActorCritic(action_space=3)
    print(
        mlp.tabulate(
            jax.random.PRNGKey(42),
            jnp.ones((1, 6)),
        )
    )

    # Create a train state:
    init_rng = jax.random.PRNGKey(42)
    learning_rate = 0.01
    momentum = 0.9
    state = create_train_state(
        module=mlp,
        rng=init_rng,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    del init_rng

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
