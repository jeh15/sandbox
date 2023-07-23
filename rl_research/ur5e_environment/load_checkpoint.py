import os
from absl import app
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import brax
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env
import orbax.checkpoint

import ur5e
import model
import model_utilities
import custom_wrapper
import visualize


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

    # Create Environment:
    episode_length = 200
    num_envs = 1
    env = create_environment(
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=False,
        batch_size=num_envs,
        backend='generalized'
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)

    # Filepath:
    filename = "models/universal_robots/scene_brax.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        filename,
    )
    pipeline_model = brax.io.mjcf.load(filepath)

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
    learning_rate = 1e-3
    end_learning_rate = 1e-6
    transition_steps = 100
    transition_begin = 400
    ppo_steps = 10

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

    target = {'model': model_state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
    model_state = orbax_checkpointer.restore(checkpoint_path, item=target)['model']

    state_history = []
    key, env_key = jax.random.split(initial_key)
    states = reset_fn(env_key)
    state_history.append(states)
    for environment_step in range(episode_length):
            key, env_key = jax.random.split(env_key)
            model_key = jax.random.split(env_key, num_envs)
            model_input = states.obs
            mean, std, values = model_utilities.forward_pass(
                model_params=model_state.params,
                apply_fn=model_state.apply_fn,
                x=model_input,
                key=model_key,
            )
            actions, log_probability, entropy = model_utilities.select_action(
                mean=mean,
                std=std,
                key=env_key,
            )
            next_states = step_fn(
                states,
                actions,
            )
            states = next_states
            state_history.append(states)

    # Squeeze states for visualization:
    formatted_state_history = []
    for state in state_history:
        state_x = state.pipeline_state.x.replace(
            pos=jnp.squeeze(state.pipeline_state.x.pos),
            rot=jnp.squeeze(state.pipeline_state.x.rot),
        )
        state_xd = state.pipeline_state.xd.replace(
            ang=jnp.squeeze(state.pipeline_state.xd.ang),
            vel=jnp.squeeze(state.pipeline_state.xd.vel),
        )
        state_pipeline_state = state.pipeline_state.replace(
            q=jnp.squeeze(state.pipeline_state.q),
            qd=jnp.squeeze(state.pipeline_state.qd),
            x=state_x,
            xd=state_xd,
        )
        state = state.replace(
            pipeline_state=state_pipeline_state,
        )
        formatted_state_history.append(state)

    video_filepath = os.path.join(
        os.path.dirname(__file__),
        "videos/ur5e_policy",
    )
    visualize.create_video(
        sys=pipeline_model,
        states=formatted_state_history,
        width=1280,
        height=720,
        name="ur5e",
        filepath=video_filepath,
        source="environment",
    )


if __name__ == '__main__':
    app.run(main)
