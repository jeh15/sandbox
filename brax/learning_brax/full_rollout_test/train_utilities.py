import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

# Imports Custom Modules:
import model_utilities

# Imports for Types:
from flax.training.train_state import TrainState
from brax.envs.env import State
from brax.envs import env as BraxEnv
from jax.random import PRNGKey


@functools.partial(jax.jit, static_argnames=['reset_fn', 'step_fn', 'episode_length'])
def rollout(
        model_state: TrainState,
        key: PRNGKey,
        reset_fn: Callable,
        step_fn: Callable,
        episode_length: int,
):
    """ Rollout of Environment Episode """
    # Generate Rollout RNG Keys:
    key, reset_key, env_key = jax.random.split(key, 3)
    # Reset Environment:
    states = reset_fn(reset_key)

    # jax.lax.scan function:
    def train_step(
            carry: Tuple[TrainState, State, PRNGKey],
            data: Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike,
                        jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
    ) -> Tuple[Tuple[TrainState, State, PRNGKey], jnp.ndarray]:
        # Unpack Carry Tuple:
        model_state, states, env_key = carry

        # Brax Environment Step:
        key, env_key = jax.random.split(env_key)
        mean, std, values = model_utilities.forward_pass(
            model_state.params,
            model_state.apply_fn,
            states.obs,
        )
        actions, log_probability, entropy = model_utilities.select_action(
            mean,
            std,
            env_key,
        )
        next_states = step_fn(
            states,
            actions,
            env_key,
        )
        states_episode = states.obs
        values_episode = jnp.squeeze(values)
        log_probability_episode = jnp.squeeze(log_probability)
        actions_episode = jnp.squeeze(actions)
        rewards_episode = jnp.squeeze(states.reward)
        masks_episode = jnp.where(states.done == 0, 1.0, 0.0)
        carry = (model_state, next_states, env_key)
        data = (
            states_episode, values_episode, log_probability_episode,
            actions_episode, rewards_episode, masks_episode,
        )
        return carry, data

    # Scan over episode:
    carry, data = jax.lax.scan(
        train_step,
        (model_state, states, env_key),
        (),
        episode_length,
    )

    # Unpack carry and data:
    model_state, states, env_key = carry
    states_episode, values_episode, log_probability_episode, \
        actions_episode, rewards_episode, masks_episode = data

    # No Gradient Calculation:
    _, _, values = jax.lax.stop_gradient(
        model_utilities.forward_pass(
            model_state.params,
            model_state.apply_fn,
            states.obs,
        ),
    )

    # Calculate Advantage:
    values_episode = jnp.concatenate(
        [values_episode, jnp.expand_dims(jnp.squeeze(values), axis=0)],
        axis=0,
    )
    advantage_episode, returns_episode = jax.lax.stop_gradient(
        model_utilities.calculate_advantage(
            rewards_episode,
            values_episode,
            masks_episode,
            episode_length,
        ),
    )

    # Update Function:
    model_state, loss = model_utilities.train_step(
        model_state,
        states_episode,
        actions_episode,
        advantage_episode,
        returns_episode,
        log_probability_episode,
    )

    return model_state, loss, carry, data
