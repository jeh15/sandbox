import os
from absl import app
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import brax
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env

import model
import ur5e
import custom_wrapper
import visualize

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


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
    env = ur5e.ur5e(**kwargs)

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
        jax.random.split(key, input_size[0]),
    )['params']
    return params


# @optax.inject_hyperparams allows introspection of learning_rate
@optax.inject_hyperparams
def optimizer(learning_rate):
    return optax.chain(
        optax.amsgrad(
            learning_rate=learning_rate,
        ),
    )


def create_train_state(module, params, optimizer):
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optimizer,
    )


def main(argv=None):
    # RNG Key:
    key_seed = 42

    best_reward = np.NINF
    best_iteration = 0

    # Create Environment:
    episode_length = 200
    num_envs = 2
    env = create_environment(
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
        backend='generalized'
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Create initial key:
    initial_key = jax.random.PRNGKey(key_seed)

    # Vmap Network:
    asset_path = r'ur5e_model/scene.xml'
    filepath = os.path.join(os.path.dirname(__file__), asset_path)
    pipeline_model = brax.io.mjcf.load(filepath)

    network = model.ActorCriticNetworkVmap(
        action_space=env.action_size,
        env=pipeline_model,
    )

    initial_params = init_params(
        module=network,
        input_size=(num_envs, env.obs_size),
        key=initial_key,
    )

    # Hyperparameters:
    learning_rate = 1e-3
    end_learning_rate = 1e-6
    transition_steps = 100
    transition_begin = 100
    ppo_steps = 10
    training_length = 500

    # Create a train state:
    schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=end_learning_rate,
        transition_steps=ppo_steps * transition_steps,
        transition_begin=ppo_steps * transition_begin,
    )
    tx = optimizer(learning_rate=schedule)
    model_state = create_train_state(
        module=network,
        params=initial_params,
        optimizer=tx,
    )
    del initial_params
    

if __name__ == '__main__':
    app.run(main)

