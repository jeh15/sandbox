from absl import app

from clu import metrics
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
import optax

import model_combined

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


def main(argv=None):
    mlp = model_combined.ActorCritic(action_space=3)
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
    combined_state = create_train_state(
        module=mlp,
        rng=init_rng,
        learning_rate=learning_rate,
        momentum=momentum,
    )

    pdb.set_trace()


if __name__ == '__main__':
    app.run(main)
