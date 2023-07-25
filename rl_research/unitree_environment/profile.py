import os
import pickle
import time
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
import model_utilities_v2 as model_utilities
import unitree_a1
import custom_wrapper

# Debug OOM Errors:
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
# os.environ['XLA_FLAGS=--xla_dump_to']='/tmp/foo'

# jax.config.update("jax_enable_x64", True)
dtype = jnp.float32


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
    env = unitree_a1.unitree_a1(**kwargs)

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
    jax.profiler.start_trace("/tmp/tensorboard")

    # RNG Key:
    key_seed = 42

    best_reward = np.NINF
    best_iteration = 0

    # Create Environment:
    episode_length = 200
    episode_train_length = 20
    num_envs = 32
    env = create_environment(
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
        backend='generalized'
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)

    # Filepath:
    filename = "models/unitree/scene.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        filename,
    )
    pipeline_model = brax.io.mjcf.load(filepath)
    pipeline_model = pipeline_model.replace(dt=0.002)

    # network = model.ActorCriticNetworkVmap(
    #     action_space=env.action_size,
    #     nodes=1,
    #     sys=pipeline_model,
    # )

    network = model.ActorCriticNetworkVmap(
        action_space=env.action_size,
        nodes=5,
        sys=pipeline_model,
    )

    initial_params = init_params(
        module=network,
        input_size=(num_envs, env.observation_size),
        key=initial_key,
    )

    # Hyperparameters:
    learning_rate = 1e-4
    end_learning_rate = 1e-6
    transition_steps = 100
    transition_begin = 100
    ppo_steps = 5

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

    # Learning Loop:
    key, env_key = jax.random.split(initial_key)
    states = reset_fn(env_key)

    key, env_key = jax.random.split(env_key)
    model_key = jax.random.split(env_key, num_envs)
    model_input = states.obs
    mean, std, values = model_utilities.forward_pass(
        model_state.params,
        model_state.apply_fn,
        model_input,
        model_key,
    )
    mean.block_until_ready()
    std.block_until_ready()
    values.block_until_ready()

    jax.profiler.save_device_memory_profile("memory.prof")


if __name__ == '__main__':
    app.run(main)
