from typing import Optional
import os
from absl import app

import jax
import jax.numpy as jnp
import brax.generalized.constraint
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env

import unitree_a1
import custom_wrapper


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


def main(argv=None):
    # Initialize System:
    filename = "models/unitree/scene.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        filename,
    )

    env = create_environment(
        episode_length=None,
        auto_reset=False,
        batch_size=False,
        backend='generalized',
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    initial_key = jax.random.PRNGKey(42)
    state = reset_fn(initial_key)

    b = state.pipeline_state.con_jac @ state.pipeline_state.mass_mx_inv @ state.pipeline_state.qf_smooth - state.pipeline_state.con_aref
    dummy_input = jnp.zeros_like(b)
    jax.jacobian(brax.generalized.constraint.force, argnums=0)(dummy_input, env.sys, state.pipeline_state)


if __name__ == "__main__":
    app.run(main)
