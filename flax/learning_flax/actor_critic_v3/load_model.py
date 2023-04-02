from absl import app
from absl import flags

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import gymnasium as gym

import model
import model_utilities


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_checkpoint',
    'server/repository/sandbox/flax/learning_flax/actor_critic_v3/checkpoints/checkpoint_0',
    'File path to PyTree checkpoint.',
)
flags.DEFINE_string(
    'environment_id',
    'CartPole-v1',
    'Environment ID to simulate.',
)
flags.DEFINE_integer(
    'episode_length',
    500,
    'Max length that an episode will play out.',
)


def create_train_state(module, params, learning_rate=0.0):
    """Creates an initial `TrainState`."""
    tx = optax.adam(
        learning_rate=learning_rate,
    )
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
    )


def load_train_state(module, params, tx):
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
    )


def init_params(module, input_size, key):
    params = module.init(
        key,
        jnp.ones(input_size),
    )['params']
    return params


def main(argv=None):
    # RNG Key:
    key_seed = 42
    key = jax.random.PRNGKey(key_seed)
    key, subkey = jax.random.split(key)

    # Create Environment:
    env = gym.make(
            FLAGS.environment_id,
            render_mode="rgb_array",
            max_episode_steps=FLAGS.episode_length,
    )
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        episode_trigger=lambda x: x == 0,
        name_prefix='pretrained-replay'
    )

    # Initize Model:
    network = model.ActorCriticNetwork(action_space=env.action_space.n)

    # Create a empty train state:
    empty_state = create_train_state(
        module=network,
        params=init_params(
            module=network,
            input_size=env.observation_space.shape[0],
            key=key,
        ),
    )

    # Load Model:
    model_state = checkpoints.restore_checkpoint(
        ckpt_dir=FLAGS.model_checkpoint,
        target=empty_state,
    )

    states, infos = env.reset()
    done = False
    while not done:
        # Forward Pass of Network:
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
        done = terminated or truncated

    env.close()


if __name__ == '__main__':
    app.run(main)
