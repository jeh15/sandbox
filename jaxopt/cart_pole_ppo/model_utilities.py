from typing import Callable, Tuple
import functools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn
import distrax


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def forward_pass(model_params, apply_fn, x):
    # Print Statement:
    print('Running Forward Pass...')
    mean, std, values = apply_fn({'params': model_params}, x)
    return mean, std, values


@jax.jit
def select_action(mean, std, key):
    probability_distribution = distrax.Normal(loc=mean, scale=std)
    actions = probability_distribution.sample(seed=key)
    log_probability = probability_distribution.log_prob(actions)
    entropy = probability_distribution.entropy()
    return actions, log_probability, entropy


@jax.jit
def evaluate_action(mean, std, action):
    probability_distribution = distrax.Normal(loc=mean, scale=std)
    log_probability = probability_distribution.log_prob(action)
    entropy = probability_distribution.entropy()
    return log_probability, entropy


# Vmap Version:
@functools.partial(jax.jit, static_argnames=['episode_length'])
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None), out_axes=(0, 0))
def calculate_advantage(rewards, values, mask, episode_length):
    gamma = 0.99
    lam = 0.95
    gae = 0.0
    advantage = []
    for i in reversed(range(episode_length)):
        error = rewards[i] + gamma * values[i+1] * mask[i] - values[i]
        gae = error + gamma * lam * mask[i] * gae
        advantage.append(gae)
    advantage = jnp.array(advantage, dtype=jnp.float64)[::-1]
    returns = advantage + values[:-1]
    return advantage, returns


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def loss_function(
    model_params,
    apply_fn,
    model_input,
    actions,
    advantages,
    returns,
    previous_log_probability,
):
    # Print Statement:
    print('Running Loss Function...')

    # Algorithm Coefficients:
    value_coeff = 0.5
    entropy_coeff = 0.01
    clip_coeff = 0.2

    # Forward Pass Rollout:
    # Change leading axis of scan arrays: redundant... maybe change test.py...
    model_input = jnp.swapaxes(
        jnp.asarray(model_input), axis1=1, axis2=0,
    )
    actions = jnp.swapaxes(
        jnp.asarray(actions), axis1=1, axis2=0,
    )
    length = model_input.shape[0]

    # This performs better?
    def forward_pass_rollout(carry, xs):
        model_input, actions = xs
        mean, std, values = forward_pass(model_params, apply_fn, model_input)
        mean, std, values = jnp.squeeze(mean), jnp.squeeze(std), jnp.squeeze(values)
        # Replay actions:
        log_probability, entropy = jax.vmap(evaluate_action)(mean, std, actions)
        carry = None
        data = (values, log_probability, entropy)
        return carry, data

    # # Parallelize instead of scan:
    # model_input = jnp.reshape(model_input, (-1, model_input.shape[-1]))
    # actions = jnp.reshape(actions, (-1, 1))
    # mean, std, values, _ = forward_pass(model_params, apply_fn, model_input)
    # log_probability, entropy = jax.vmap(evaluate_action)(mean, std, actions)

    # # Needs to go back to (batch_size, episode_length)
    # mean = jnp.reshape(mean, (length, -1))
    # std = jnp.reshape(std, (length, -1))
    # values = jnp.reshape(values, (length, -1))
    # log_probability = jnp.reshape(log_probability, (length, -1))
    # entropy = jnp.reshape(entropy, (length, -1))

    # Scan over replay:
    carry, data = jax.lax.scan(
        forward_pass_rollout,
        None,
        (model_input, actions),
        length,
    )
    values, log_probability, entropy = data
    values = jnp.swapaxes(
        jnp.asarray(values), axis1=1, axis2=0,
    )
    log_probability = jnp.swapaxes(
        jnp.asarray(log_probability), axis1=1, axis2=0,
    )
    entropy = jnp.swapaxes(
        jnp.asarray(entropy), axis1=1, axis2=0,
    )

    # Calculate Ratio: (Should this be No Grad?)
    log_ratios = log_probability - previous_log_probability
    ratios = jnp.exp(log_ratios)

    # Policy Loss:
    unclipped_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(
        1.0 - clip_coeff,
        ratios,
        1.0 + clip_coeff,
    )
    ppo_loss = -jnp.mean(
        jnp.minimum(unclipped_loss, clipped_loss),
    )

    # Value Loss:
    value_loss = value_coeff * jnp.mean(
        jnp.square(values - returns),
    )

    # Entropy Loss:
    entropy_loss = -entropy_coeff * jnp.mean(entropy)

    return ppo_loss + value_loss + entropy_loss


@jax.jit
def train_step(
    model_state,
    model_input,
    actions,
    advantages,
    returns,
    previous_log_probability,
):
    # Print Statement:
    print('Running Train Step...')

    gradient_function = jax.value_and_grad(loss_function)
    loss, gradients = gradient_function(
        model_state.params,
        model_state.apply_fn,
        model_input,
        actions,
        advantages,
        returns,
        previous_log_probability,
    )
    model_state = model_state.apply_gradients(grads=gradients)
    return model_state, loss
