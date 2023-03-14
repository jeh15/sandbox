import functools
from typing import Any, Callable, Tuple, List

from absl import logging
import flax
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.random
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float,
):
    """Use Generalized Advantage Estimation (GAE) to compute advantages.
    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.
    Args:
        rewards: array shaped (actor_steps, num_agents), rewards from the game
        terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                        and ones for non-terminal states
        values: array shaped (actor_steps, num_agents), values estimated by critic
        discount: RL discount usually denoted with gamma
        gae_param: GAE parameter usually denoted with lambda
    Returns:
        advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    assert rewards.shape[0] + 1 == values.shape[0], ('One more value needed; Eq. '
                                                    '(12) in PPO paper requires '
                                                    'V(s_{t+1}) for delta_t')
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff
        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


def loss_fn(
    params: flax.core.FrozenDict,
    policy_action: Callable[..., Any],
    minibatch: Tuple,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
    """Evaluate the loss function.
    Compute loss as a sum of three components: the negative of the PPO clipped
    surrogate objective, the value function loss and the negative of the entropy
    bonus.
    Args:
        params: the parameters of the actor-critic model
        apply_fn: the actor-critic model's apply function
        minibatch: Tuple of five elements forming one experience batch:
                states: shape (batch_size, 84, 84, 4)
                actions: shape (batch_size, 84, 84, 4)
                old_log_probs: shape (batch_size,)
                returns: shape (batch_size,)
                advantages: shape (batch_size,)
        clip_param: the PPO clipping parameter used to clamp ratios in loss function
        vf_coeff: weighs value function loss in total loss
        entropy_coeff: weighs entropy bonus in the total loss
    Returns:
        loss: the PPO loss, scalar quantity
    """
    states, actions, old_log_probs, returns, advantages = minibatch
    log_probs, values = policy_action(states)
    values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
    probs = jnp.exp(log_probs)

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)

    entropy = jnp.sum(-probs*log_probs, axis=1).mean()

    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1. - clip_param, ratios, 1. + clip_param)
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

    return ppo_loss + vf_coeff*value_loss - entropy_coeff*entropy


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: train_state.TrainState,
    trajectories: Tuple,
    batch_size: int,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float
):
    """Compilable train step.
    Runs an entire epoch of training (i.e. the loop over minibatches within
    an epoch is included here for performance reasons).
    Args:
        state: the train state
        trajectories: Tuple of the following five elements forming the experience:
                    states: shape (steps_per_agent*num_agents, 84, 84, 4)
                    actions: shape (steps_per_agent*num_agents, 84, 84, 4)
                    old_log_probs: shape (steps_per_agent*num_agents, )
                    returns: shape (steps_per_agent*num_agents, )
                    advantages: (steps_per_agent*num_agents, )
        batch_size: the minibatch size, static argument
        clip_param: the PPO clipping parameter used to clamp ratios in loss function
        vf_coeff: weighs value function loss in total loss
        entropy_coeff: weighs entropy bonus in the total loss
    Returns:
        optimizer: new optimizer after the parameters update
        loss: loss summed over training steps
    """
    iterations = trajectories[0].shape[0] // batch_size
    trajectories = jax.tree_util.tree_map(
        lambda x: x.reshape((iterations, batch_size) + x.shape[1:]),
        trajectories,
    )
    loss = 0.0
    for batch in zip(*trajectories):
        grad_fn = jax.value_and_grad(loss_fn)
        l, grads = grad_fn(
            state.params,
            state.apply_fn,
            batch,
            clip_param,
            vf_coeff,
            entropy_coeff,
        )
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss
